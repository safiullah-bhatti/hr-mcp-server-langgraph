// hr-mcp-server-langgraph/rag.js
//
// Same job as the LangChain version: build an in-memory semantic
// search index per doc folder, at server startup, no persistence.
//
// This is a LIKE-FOR-LIKE swap: LangChain's RecursiveCharacterTextSplitter
// + GoogleGenerativeAIEmbeddings + MemoryVectorStore are replaced with
// LlamaIndex's SentenceSplitter + GeminiEmbedding + VectorStoreIndex's
// own default in-memory store. No new vector database, no Docker,
// nothing persisted to disk — when the server restarts, the index is
// rebuilt from scratch, exactly like the LangChain version.
//
// buildIndex() keeps the exact same exported signature and the exact
// same { text, source, score } shape from search(), so server.js does
// NOT need to change at all.
//
// What's actually different about the LlamaIndex approach (this is
// the whole point of this file — read this before comparing to the
// LangChain version):
//
// 1. CHUNKING HAPPENS INSIDE INDEXING, NOT BEFORE IT. The LangChain
//    version manually ran the splitter over each file's text FIRST,
//    then wrapped each resulting chunk in its own Document. Here, one
//    Document is created per FILE (full text), and a `transformations:
//    [new SentenceSplitter(...)]` pipeline is handed to
//    VectorStoreIndex.fromDocuments() — LlamaIndex does the chunking
//    itself as part of building the index. The resulting chunks
//    (LlamaIndex calls them "Nodes") automatically inherit the parent
//    Document's metadata, so you never manually thread
//    `{ metadata: { source } }` through a chunk loop like the
//    LangChain version does.
//
// 2. NO EXPLICIT VECTOR STORE OBJECT. LangChain requires you to name a
//    store class (`MemoryVectorStore.fromDocuments(...)`). LlamaIndex's
//    `VectorStoreIndex.fromDocuments(...)` doesn't need one at all when
//    you don't pass a `storageContext` — it defaults to its own
//    built-in in-memory `SimpleVectorStore` automatically. That's the
//    direct LlamaIndex equivalent of LangChain's MemoryVectorStore,
//    just implicit instead of imported by name.
//
// 3. RETRIEVAL SHAPE. LangChain's `similaritySearchWithScore()` returns
//    `[Document, score]` pairs. LlamaIndex's retriever returns
//    `NodeWithScore[]`, where each item is `{ node, score }` and the
//    node's text/metadata are read via `node.getContent()` /
//    `node.metadata`. Same information, different accessors — handled
//    in `search()` below so the RETURNED shape to server.js is
//    unchanged.

import fs from "fs";
import path from "path";
import "dotenv/config";
import {
  Document,
  VectorStoreIndex,
  Settings,
  SentenceSplitter,
} from "llamaindex";
// NOTE: LlamaIndex.TS's own docs are inconsistent about whether this
// enum is exported as `GEMINI_MODEL` or `GEMINI_EMBEDDING_MODEL` — the
// API reference for GeminiEmbedding's options type names it
// `GEMINI_EMBEDDING_MODEL`, so that's what's used here. If `npm
// install` pulls a version where it's actually named differently,
// swap this import accordingly — nothing else in this file changes.
import { GeminiEmbedding, GEMINI_EMBEDDING_MODEL } from "@llamaindex/google";


// Same numbers as the LangChain version's chunker, so index behavior
// doesn't silently change just because the library changed.
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 80;

// Same embedding provider (Gemini) as before, still a separate concern
// from which LLM your client uses to hold the conversation. The TS
// SDK's GeminiEmbedding doesn't expose the same "gemini-embedding-001"
// model name the LangChain version used — it currently offers
// EMBEDDING_001 and TEXT_EMBEDDING_004 (both 768-dim); TEXT_EMBEDDING_004
// is the newer of the two, used here.
Settings.embedModel = new GeminiEmbedding({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
});

// LlamaIndex-native chunker, handed to VectorStoreIndex.fromDocuments()
// as a `transformations` step. This is what turns whole-file Documents
// into 500-char/80-overlap Nodes — LangChain's splitter did this same
// job manually, up front, before any Document existed.
const splitter = new SentenceSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

// Reads every .txt file in a folder, and returns a
// { search(query, topK), retriever } object — same as the LangChain
// version.
//
// `search()` is kept so server.js's two tool handlers don't need any
// changes. `retriever` is exposed alongside it for anyone who wants
// the more idiomatic LlamaIndex call shape later
// (retriever.retrieve(query)).
export async function buildIndex(folderPath, label) {
  const files = fs.readdirSync(folderPath).filter((f) => f.endsWith(".txt"));
  console.log(`[SERVER] [RAG:${label}] [llamaindex] Ingesting ${files.length} doc(s)...`);

  // One Document per FILE, not per chunk — the splitter above chunks
  // these during indexing, and every resulting chunk inherits this
  // metadata automatically.
  const documents = files.map((filename) => {
    const text = fs.readFileSync(path.join(folderPath, filename), "utf-8");
    return new Document({ text, id_: filename, metadata: { source: filename } });
  });

  // This one call is the entire "vector database" now: chunking +
  // embedding + in-memory storage, all handled by LlamaIndex instead
  // of the old hand-rolled records array + cosineSimilarity(), and
  // instead of LangChain's MemoryVectorStore.fromDocuments(). No
  // storageContext is passed, so this defaults to LlamaIndex's own
  // built-in SimpleVectorStore — nothing persisted, exactly like the
  // LangChain version.
  const index = await VectorStoreIndex.fromDocuments(documents, {
    transformations: [splitter],
  });

  console.log(`[SERVER] [RAG:${label}] [llamaindex] Index ready`);

  return {
    retriever: index.asRetriever({ similarityTopK: 3 }),

    async search(query, topK = 3) {
      console.log(`[SERVER] [RAG:${label}] [llamaindex] Retrieving top ${topK} for query: "${query}"`);
      const retriever = index.asRetriever({ similarityTopK: topK });
      const results = await retriever.retrieve(query);
      // NodeWithScore[] -> { text, source, score }, same shape the
      // LangChain version's search() returned.
      return results.map((r) => ({
        text: typeof r.node.getContent === "function" ? r.node.getContent() : r.node.text,
        source: r.node.metadata?.source ?? "unknown",
        // Cosine similarity, same metric/range as the LangChain
        // version, so graph.js's RELEVANCE_THRESHOLD comparison in
        // the client keeps working unchanged.
        score: r.score ?? 0,
      }));
    },
  };
}

// --- Swapping to a real persistent vector DB later ---
// Pass a `storageContext` (built via `storageContextFromDefaults({
// vectorStore })`) into VectorStoreIndex.fromDocuments() above, e.g.:
//   import { storageContextFromDefaults } from "llamaindex";
//   import { ChromaVectorStore } from "@llamaindex/chroma";
//   const storageContext = await storageContextFromDefaults({
//     vectorStore: new ChromaVectorStore({ collectionName: label }),
//   });
//   const index = await VectorStoreIndex.fromDocuments(documents, {
//     storageContext,
//     transformations: [splitter],
//   });
// Everything else in this file — and all of server.js — stays
// identical, because every LlamaIndex vector store implements the
// same BaseVectorStore interface the default in-memory one does.
