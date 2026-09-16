// hr-mcp-server-langchain/rag.js
//
// Same job as the original rag.js: build an in-memory semantic search
// index per doc folder, at server startup, no persistence.
//
// What changed vs. the hand-rolled version: chunking, embedding, and
// the vector store + cosine-similarity search are no longer written by
// hand — they're LangChain's RecursiveCharacterTextSplitter,
// GoogleGenerativeAIEmbeddings, and MemoryVectorStore.
//
// Why this matters even though the *behavior* is the same: buildIndex()
// keeps the exact same exported signature and the exact same
// { text, source, score } shape from search(), so server.js does NOT
// need to change at all. The only thing that changes is what's INSIDE
// buildIndex. And because MemoryVectorStore implements the same
// VectorStore interface as Chroma/PGVectorStore, swapping to a real
// persistent vector DB later is a ~3-line change in this file only —
// see the comment at the bottom.

import fs from "fs";
import path from "path";
import "dotenv/config";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";

// Same numbers as the original hand-rolled chunker, so index behavior
// doesn't silently change when you swap in this version.
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 80;

// Same embedding provider (Gemini) as before — this is still a
// SEPARATE concern from which LLM your client uses to hold the
// conversation. The server can embed with Gemini while the client
// talks to Claude; nothing ties these together.
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "text-embedding-004",
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

// Reads every .txt file in a folder, chunks + embeds each one, and
// returns a { search(query, topK), retriever } object.
//
// `search()` is kept so server.js's two tool handlers don't need any
// changes. `retriever` is exposed alongside it for anyone who wants to
// move to the more idiomatic LangChain call shape later
// (retriever.invoke(query)) — e.g. once this gets bound into a
// LangGraph router down the line.
export async function buildIndex(folderPath, label) {
  const files = fs.readdirSync(folderPath).filter((f) => f.endsWith(".txt"));
  console.log(`[SERVER] [RAG:${label}] Ingesting ${files.length} doc(s)...`);

  const docs = [];
  for (const filename of files) {
    const text = fs.readFileSync(path.join(folderPath, filename), "utf-8");
    const chunks = await splitter.splitText(text);
    for (const chunk of chunks) {
      docs.push(new Document({ pageContent: chunk, metadata: { source: filename } }));
    }
  }

  // This one line is the entire "vector database" now: chunking +
  // embedding + storage, all handled by LangChain instead of the old
  // hand-rolled records array + cosineSimilarity().
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

  console.log(`[SERVER] [RAG:${label}] Indexed ${docs.length} chunk(s) total`);

  return {
    retriever: store.asRetriever({ k: 3 }),

    async search(query, topK = 3) {
      // similaritySearchWithScore returns [Document, score] pairs,
      // score = cosine similarity (same metric the hand-rolled
      // version used), highest first — same shape as before.
      const results = await store.similaritySearchWithScore(query, topK);
      return results.map(([doc, score]) => ({
        text: doc.pageContent,
        source: doc.metadata.source,
        score,
      }));
    },
  };
}

// --- Swapping to a real persistent vector DB later ---
// Replace the MemoryVectorStore import + the one `fromDocuments` line
// above with, e.g.:
//   import { Chroma } from "@langchain/community/vectorstores/chroma";
//   const store = await Chroma.fromDocuments(docs, embeddings, {
//     collectionName: label,
//     url: process.env.CHROMA_URL,
//   });
// Everything else in this file — and all of server.js — stays
// identical, because both stores implement the same VectorStore
// interface (.asRetriever(), .similaritySearchWithScore()).
//
// IMPORTANT (re: "should hr-policy/engineering/admin each get a
// different vector DB engine?"): buildIndex() is called once per
// doc folder (see server.js — hr-policy, engineering, and now
// admin-policies each get their own call). Nothing stops each call
// from using a *different* store — e.g. hr-policy on Chroma,
// engineering on PGVectorStore, admin-policies on Qdrant — since
// they're three independent indices already, each with its own
// buildIndex() call and its own tool. You'd do that for
// infra/ops reasons (who hosts what, access control, existing
// team expertise), not because LangChain requires it and NOT
// because it changes anything about whether you need LangGraph —
// that question is about how the *tool-calling loop* is
// orchestrated on the client, which is a completely separate layer
// from which database sits behind any one RAG tool. Swap the
// `MemoryVectorStore.fromDocuments(...)` line per-domain if/when
// each domain gets its own real deployment; the { text, source,
// score } contract server.js and the client depend on stays fixed
// either way.
