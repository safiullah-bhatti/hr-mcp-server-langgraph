// hr-mcp-server-langgraph/rag.js
//
// LlamaIndex.TS version. Same job as the LangChain version: build a
// semantic-search index per doc folder, at server startup, and expose
// { search(query, topK), retriever } so server.js does NOT need to
// change at all — same exported buildIndex() signature, same
// { text, source, score } shape out of search().
//
// What's actually different from the LangChain version (this is the
// point of this file):
//
// 1. CHUNKING. LangChain's version manually split each file's text
//    into chunks with RecursiveCharacterTextSplitter BEFORE creating
//    Documents — so LangChain only ever saw pre-chunked text.
//    LlamaIndex flips this: you hand it one Document per FILE (full
//    text + metadata), and a `transformations: [new SentenceSplitter(...)]`
//    pipeline does the chunking for you as documents are ingested into
//    the index. The resulting chunks (LlamaIndex calls them "Nodes")
//    automatically inherit the parent Document's metadata (source
//    filename) — you never have to thread `{ metadata: { source } }`
//    through a manual loop like the LangChain version did.
//
// 2. VECTOR STORE PER DOMAIN. This is the other half of what you
//    asked for: each of the three RAG domains now gets a genuinely
//    different, persistent vector database instead of one shared
//    in-memory store:
//      - hr-policy        -> ChromaDB   (@llamaindex/chroma)
//      - engineering       -> Postgres/pgvector (@llamaindex/postgres)
//      - admin-policies    -> Qdrant     (@llamaindex/qdrant)
//    All three implement LlamaIndex's same BaseVectorStore interface,
//    so buildIndex() itself doesn't care which one it's talking to —
//    only vectorStoreFor(label) below knows. That's the LlamaIndex
//    equivalent of the "swap the store, nothing else changes" comment
//    that was already in the old rag.js.
//
// CAVEAT (read before demoing): unlike the old in-memory store, these
// three are persistent. Every time you restart `node server.js`, the
// docs get re-embedded and re-inserted into whatever collection/table
// already exists — Chroma and Postgres will happily accumulate
// duplicate chunks across restarts. For a demo this doesn't break
// anything (you'll just get repeated matches with the same score),
// but if you want a clean slate, run `docker compose down -v` between
// runs to wipe the volumes. A production version would check
// "has this domain already been indexed?" before re-inserting —
// intentionally left out here to keep this file focused on the
// LlamaIndex swap itself.
//
// DEPRECATION NOTE: @llamaindex/qdrant is currently marked deprecated
// on npm (no longer maintained by the LlamaIndex team, but still
// published and functional as of this writing). It's used here
// because it's still the most direct way to plug Qdrant into
// LlamaIndex.TS's VectorStoreIndex. If it stops working after an
// LlamaIndex core upgrade, the fix is to implement a small custom
// class extending LlamaIndex's BaseVectorStore against
// @qdrant/js-client-rest directly — everything else in this file
// (buildIndex, search, the SentenceSplitter pipeline) stays the same.

import fs from "fs";
import path from "path";
import "dotenv/config";
import { PDFParse } from "pdf-parse";
import {
  Document,
  VectorStoreIndex,
  Settings,
  SentenceSplitter,
  storageContextFromDefaults,
} from "llamaindex";
// NOTE: LlamaIndex.TS's own docs are inconsistent about whether this
// enum is exported as `GEMINI_MODEL` or `GEMINI_EMBEDDING_MODEL` —
// the API reference for GeminiEmbedding's options type names it
// `GEMINI_EMBEDDING_MODEL`, so that's what's used here. If `npm
// install` pulls a version where it's actually named differently,
// swap this import accordingly — everything else in this file is
// unaffected either way.
import { GeminiEmbedding } from "@llamaindex/google";
import { ChromaVectorStore } from "@llamaindex/chroma";
import { PGVectorStore } from "@llamaindex/postgres";
import { QdrantVectorStore } from "@llamaindex/qdrant";

// Same numbers as the original hand-rolled/LangChain chunker, so
// retrieval granularity doesn't silently change just because the
// library changed.
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 80;

// GEMINI_MODEL.TEXT_EMBEDDING_004 is Google's newer embedding model,
// 768-dimensional. (The TS SDK's GeminiEmbedding currently exposes
// two models — EMBEDDING_001 and TEXT_EMBEDDING_004 — both 768-dim;
// unlike LangChain's gemini-embedding-001 usage in the old version,
// the TS SDK doesn't expose a configurable outputDimensionality, so
// we pick the fixed 768-dim model and use that dimension everywhere
// a store needs to know it up front, e.g. Postgres below.)
// GeminiEmbedding sends no reduced outputDimensionality in its
// embedContent request, so gemini-embedding-001 returns the default
// 3072-dimensional vectors. Postgres needs this size for its vector column.
const EMBED_DIM = 3072;

Settings.embedModel = new GeminiEmbedding({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
});

// The chunking pipeline every domain's index is built with. Handed to
// VectorStoreIndex.fromDocuments() as `transformations` — this is
// what turns whole-file Documents into 500-char/80-overlap Nodes.
const splitter = new SentenceSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

// ---- Per-domain vector store selection ----
// This function is the entire "which real vector DB does this domain
// use" decision. Everything else in buildIndex() is store-agnostic.
function vectorStoreFor(label) {
  switch (label) {
    case "hr-policy":
      console.log(`[SERVER] [RAG:${label}] backing store: ChromaDB (${process.env.CHROMA_URL || "http://localhost:8000"})`);
      return new ChromaVectorStore({
        collectionName: "hr_policy",
        host: process.env.CHROMA_URL || "http://localhost:8000",
      });

    case "engineering":
      console.log(`[SERVER] [RAG:${label}] backing store: Postgres/pgvector`);
      return new PGVectorStore({
        // The current LlamaIndex Postgres adapter accepts pg's
        // ClientConfig under `clientConfig`; a top-level
        // `connectionString` is ignored, leaving `config.client`
        // undefined and causing a constructor error.
        clientConfig: {
          connectionString:
            process.env.PG_CONNECTION_STRING ||
            "postgresql://llamaindex:llamaindex@localhost:5433/engineering_docs",
        },
        // A prior run may have created this table with VECTOR(768);
        // CREATE TABLE IF NOT EXISTS won't change an existing column.
        tableName: "engineering_practices_3072",
        dimensions: EMBED_DIM,
      });

    case "admin-policies":
      console.log(`[SERVER] [RAG:${label}] backing store: Qdrant (${process.env.QDRANT_URL || "http://localhost:6333"})`);
      return new QdrantVectorStore({
        url: process.env.QDRANT_URL || "http://localhost:6333",
        collectionName: "admin_policies",
      });

    default:
      // Fail loudly rather than silently falling back to some default
      // store — a 4th domain added later needs an explicit decision
      // made here, same as the old file's comment already called out.
      throw new Error(
        `[SERVER] [RAG] No vector store configured for domain "${label}". Add a case in vectorStoreFor().`
      );
  }
}

// Reads every .txt and .pdf file in a folder, wraps each full file as
// one Document (metadata: source filename), and lets VectorStoreIndex
// do the chunking + embedding + storage via the domain's vector store.
//
// `search()` is kept so server.js's tool handlers don't need any
// changes. `retriever` is exposed alongside it, same as before, for
// anyone who wants the more idiomatic `retriever.retrieve(query)`
// call shape.
export async function buildIndex(folderPath, label) {
  const files = fs.readdirSync(folderPath).filter((f) => /\.(txt|pdf)$/i.test(f));
  console.log(`[SERVER] [RAG:${label}] Ingesting ${files.length} doc(s)...`);

  const documents = await Promise.all(files.map(async (filename) => {
    const filePath = path.join(folderPath, filename);
    let text;
    if (path.extname(filename).toLowerCase() === ".pdf") {
      const parser = new PDFParse({ data: fs.readFileSync(filePath) });
      try {
        text = (await parser.getText()).text;
      } finally {
        await parser.destroy();
      }
    } else {
      text = fs.readFileSync(filePath, "utf-8");
    }
    // One Document per FILE, not per chunk — LlamaIndex's own
    // SentenceSplitter transformation (passed below) does the
    // chunking, and every resulting chunk inherits this metadata.
    return new Document({ text, id_: filename, metadata: { source: filename } });
  }));

  const vectorStore = vectorStoreFor(label);
  const storageContext = await storageContextFromDefaults({ vectorStore });

  console.log(`[SERVER] [RAG:${label}] [llamaindex] building VectorStoreIndex (chunking + embedding + upsert)...`);
  const index = await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
    transformations: [splitter],
  });
  console.log(`[SERVER] [RAG:${label}] [llamaindex] index ready`);

  return {
    retriever: index.asRetriever({ similarityTopK: 3 }),

    async search(query, topK = 3) {
      console.log(`[SERVER] [RAG:${label}] [llamaindex] retrieving top ${topK} for query: "${query}"`);
      const retriever = index.asRetriever({ similarityTopK: topK });
      const results = await retriever.retrieve(query);
      return results.map((r) => ({
        text: typeof r.node.getContent === "function" ? r.node.getContent() : r.node.text,
        source: r.node.metadata?.source ?? "unknown",
        // LlamaIndex returns cosine similarity in [0,1] for these
        // stores, same metric/range the LangChain version used, so
        // graph.js's RELEVANCE_THRESHOLD comparison in the client
        // keeps working unchanged.
        score: r.score ?? 0,
      }));
    },
  };
}
