// hr-mcp-server-langgraph/rag.js
//
// Two things live in this file now, both from LlamaIndex, and they
// exist for genuinely different jobs:
//
//   buildIndex(folderPath, label)        -> "find the fact"
//     Same as before: VectorStoreIndex + embedding similarity search.
//     Good when a question is about ONE specific detail — it only
//     ever looks at the handful of chunks most similar to the query,
//     so it's fast and cheap, but it can silently miss content that
//     didn't happen to be semantically close to how the question was
//     phrased.
//
//   buildSummaryIndex(folderPath, label) -> "read everything"
//     NEW. SummaryIndex + asQueryEngine({ responseMode: "tree_summarize" }).
//     No embedding similarity step at all — it reads every single
//     chunk in the folder, unconditionally, and has the LLM
//     synthesize them together (in batches, then recursively
//     summarizing those summaries — hence "tree"). Slower and more
//     LLM calls, but it can't skip a file the way similarity search
//     can.
//
// Both return an object with the SAME shape they always would've had
// (search()/summarize() are plain async functions server.js calls),
// so nothing about how server.js is structured needs to change to use
// either one — you just call whichever fits the question.
//
// One new requirement this brings: query SYNTHESIS (actually writing
// an answer, not just finding matching chunks) needs an LLM, not just
// an embedding model. That's why Settings.llm is now set below,
// alongside the existing Settings.embedModel — two different jobs,
// both currently pointed at Gemini, both using the same API key.

import fs from "fs";
import path from "path";
import "dotenv/config";
import { PDFParse } from "pdf-parse";
import {
  Document,
  VectorStoreIndex,
  SummaryIndex,
  Settings,
  SentenceSplitter,
} from "llamaindex";
// NOTE: LlamaIndex.TS's docs aren't fully consistent about exact
// export names across versions. As documented, embeddings use the
// `GEMINI_EMBEDDING_MODEL` enum and the LLM uses a separate `gemini()`
// factory + `GEMINI_MODEL` enum (yes, two different enums — one per
// job). If `npm install` resolves a version where a name differs
// slightly, this is the one spot to adjust; nothing else in this file
// depends on the exact name.
import { GeminiEmbedding, gemini, GEMINI_MODEL_INFO_MAP } from "@llamaindex/google";

// Same numbers as before — only affects buildIndex()'s chunking;
// buildSummaryIndex() chunks the same way but reads every resulting
// chunk instead of searching them.
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 80;

// Embeddings: turns text into vectors, for similarity search
// (buildIndex only).
Settings.embedModel = new GeminiEmbedding({
  apiKey: process.env.GEMINI_API_KEY,
  model: process.env.GEMINI_EMBEDDING_MODEL || 'gemini-embedding-001',
});

// LLM: actually writes prose answers, for query synthesis
// (buildSummaryIndex only — buildIndex/search() never calls this,
// it just returns raw matched chunks).
const SUMMARY_MODEL = process.env.GEMINI_MODEL || "gemini-3.6-flash";

// The installed @llamaindex/google version does not list Gemini 3.6
// Flash in its model metadata. LlamaIndex reads contextWindow while
// constructing the summary query engine, so add Google's documented
// input limit when this adapter version does not already know the model.

console.log(`[SERVER] [RAG] [llamaindex] Using GEMINI_MODEL "${SUMMARY_MODEL}" for query synthesis (summary index).`);
if (!GEMINI_MODEL_INFO_MAP[SUMMARY_MODEL]) {
  console.log(`[SERVER] [RAG] [llamaindex] GEMINI_MODEL "${SUMMARY_MODEL}" is not in GEMINI_MODEL_INFO_MAP, adding contextWindow info for this model.`);
  const knownContextWindows = {
    "gemini-3.6-flash": 1_048_576,
  };
  const contextWindow = knownContextWindows[SUMMARY_MODEL];
  if (!contextWindow) {
    throw new Error(
      `GEMINI_MODEL "${SUMMARY_MODEL}" is not supported by the installed @llamaindex/google adapter and has no configured context window.`
    );
  }
  GEMINI_MODEL_INFO_MAP[SUMMARY_MODEL] = { contextWindow };
}

Settings.llm = gemini({
  apiKey: process.env.GEMINI_API_KEY,
  model: SUMMARY_MODEL,
});

const splitter = new SentenceSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

// Shared by both index types below — reads .txt and .pdf files from
// folder into one Document each (full file text; chunking happens
// later, during indexing, via the `transformations: [splitter]`
// option both fromDocuments() calls use).
async function loadDocuments(folderPath) {
  const files = fs
    .readdirSync(folderPath)
    .filter((filename) => /\.(txt|pdf)$/i.test(filename))
    .sort((a, b) => a.localeCompare(b));

  const documents = await Promise.all(files.map(async (filename) => {
    const filePath = path.join(folderPath, filename);
    let text;

    console.log(`[SERVER] [RAG] [llamaindex] Loading file: ${filename}`);
    if (path.extname(filename).toLowerCase() === ".pdf") {
      const parser = new PDFParse({ data: fs.readFileSync(filePath) });
      try {
        text = (await parser.getText()).text;
        console.log(`[SERVER] [RAG] [llamaindex] Extracted text from PDF: ${filename} (${text.length} chars)`);
      } finally {
        await parser.destroy();
      }

      if (!text?.trim()) {
        console.error(`[SERVER] [RAG] [llamaindex] Warning: PDF contains no extractable text: ${filename}. Scanned/image-only PDFs need OCR before ingestion.`);
        throw new Error(
          `PDF contains no extractable text: ${filename}. Scanned/image-only PDFs need OCR before ingestion.`
        );
      }
    } else {
      text = fs.readFileSync(filePath, "utf-8");
      console.log(`[SERVER] [RAG] [llamaindex] Loaded text from TXT: ${filename} (${text.length} chars)`);
    }

    return new Document({ text, id_: filename, metadata: { source: filename } });
  }));

  return { files, documents };
}

// ---- "find the fact": similarity search, unchanged from before ----
export async function buildIndex(folderPath, label) {
  const { files, documents } = await loadDocuments(folderPath);
  console.log(`[SERVER] [RAG:${label}] [buildIndex] [llamaindex] Ingesting ${files.length} doc(s) for search...`);

  const index = await VectorStoreIndex.fromDocuments(documents, {
    transformations: [splitter],
  });
  console.log(`[SERVER] [RAG:${label}] [buildIndex] [llamaindex] Search index ready`);

  return {
    retriever: index.asRetriever({ similarityTopK: 3 }),

    async search(query, topK = 3) {
      console.log(`[SERVER] [RAG:${label}] [buildIndex] [llamaindex] search() retrieving top ${topK} for: "${query}"`);
      const retriever = index.asRetriever({ similarityTopK: topK });
      const results = await retriever.retrieve(query);
      return results.map((r) => ({
        text: typeof r.node.getContent === "function" ? r.node.getContent() : r.node.text,
        source: r.node.metadata?.source ?? "unknown",
        score: r.score ?? 0,
      }));
    },
  };
}

// ---- "read everything": NEW, whole-folder summarization ----
export async function buildSummaryIndex(folderPath, label) {
  const { files, documents } = await loadDocuments(folderPath);
  console.log(`[SERVER] [RAG:${label}] [buildSummaryIndex] [llamaindex] Ingesting ${files.length} doc(s) for summarization...`);

  const index = await SummaryIndex.fromDocuments(documents, {
    transformations: [splitter],
  });

  console.log(`[SERVER] [RAG:${label}] [buildSummaryIndex] [llamaindex] Summary index built, preparing query engine...`);
  // tree_summarize: batch chunks to fit the LLM's context window,
  // summarize each batch, then recursively summarize those summaries
  // until one answer is left. This is what makes it actually cover
  // ALL the chunks instead of just the closest-matching few.
  const queryEngine = index.asQueryEngine({ responseMode: "tree_summarize" });
  console.log(`[SERVER] [RAG:${label}] [buildSummaryIndex] [llamaindex] Summary index ready (${files.length} doc(s), all chunks in scope)`);

  return {
    queryEngine,

    async summarize(query = "Summarize this collection of documents.") {
      console.log(`[SERVER] [RAG:${label}] [buildSummaryIndex] [llamaindex] summarize() reading ALL ${files.length} doc(s) for: "${query}"`);
      const response = await queryEngine.query({ query });
      return {
        text: response.toString(),
        sourceCount: files.length,
      };
    },
  };
}
