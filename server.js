// hr-mcp-server-langchain/server.js
//
// STANDALONE HTTP MCP server — runs completely independently now. Start
// it once (`node server.js`), it keeps running, and ANY MCP client
// (this project's, or a totally different app) can connect to it over
// the network. No client spawns this anymore.
//
// Secrets live in THIS server's own .env file — never passed in by,
// or visible to, any client. This is the real production pattern.
//
// LangChain note: buildIndex() and *Index.search() still keep the
// exact same signature/return shape described in rag.js — only its
// internals are LangChain (text splitter, Gemini embeddings,
// MemoryVectorStore). Two things ARE new in this version, both for
// the LangGraph client (see hr-mcp-client-langchain/graph.js):
//
// 1. A THIRD RAG tool, search_admin_docs, indexing docs/admin-policies
//    (IT/security, expenses). This is just "call buildIndex() a third
//    time and register a third tool" — adding a new document domain
//    never needed LangGraph, it works exactly the same way the first
//    two domains did with the plain tool-call loop.
//
// 2. The three RAG tools now return a small JSON envelope
//    ({ topScore, matches: [...] }) instead of a flat text blob. The
//    text itself is unchanged (still "[Source: x]\n<chunk>" per
//    match) — topScore is added ON TOP so a client-side graph can
//    make a *system-level* decision ("this retrieval was weak, go
//    retry with a reformulated query") instead of relying on the LLM
//    to notice and re-call the tool on its own. This is the one
//    concrete case in this demo where the client genuinely benefits
//    from LangGraph over a flat while-loop — see the comment block at
//    the top of graph.js for why.
//
// 3. FIX: a single McpServer instance can only be connected to ONE
//    transport at a time — connecting a second one before the first
//    fully closes throws "Already connected to a transport." A
//    LangGraph run makes more sequential tool calls per session
//    (original + retry) than the old single-call version did, which
//    made this pre-existing bug easy to hit. Fix: createMcpServer()
//    below builds a FRESH, cheap McpServer (tools re-registered
//    against the already-built indices — no re-embedding) for EVERY
//    incoming /mcp request, instead of reusing one shared instance.

import "dotenv/config";                 // MUST be first: loads .env before anything below reads process.env
import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import path from "path";
import { fileURLToPath } from "url";
import { buildIndex } from "./rag.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;

const PROJECT_NAMES = [
  "Phoenix", "Atlas", "Nimbus", "Falcon", "Orion",
  "Titan", "Nova", "Zephyr", "Quartz", "Comet",
];
const MANAGERS = ["Hassan Raza", "Ayesha Khan", "Bilal Ahmed", "Sara Malik"];

function randInt(max) {
  return Math.floor(Math.random() * (max + 1));
}

// Shared by all three RAG tool handlers — kept in one place so the
// "what does a RAG tool return" contract only has one definition.
function ragToolResult(label, matches) {
  matches.forEach((m) =>
    console.log(`[SERVER] [RAG:${label}]   match (score ${m.score.toFixed(3)}) from ${m.source}`)
  );
  const topScore = matches.length ? Math.max(...matches.map((m) => m.score)) : 0;
  const payload = {
    topScore,
    matches: matches.map((m) => ({ source: m.source, text: m.text })),
  };
  return { content: [{ type: "text", text: JSON.stringify(payload) }] };
}

// Builds a brand-new McpServer with all 7 tools registered, closing
// over the already-built indices (cheap — no re-embedding, just new
// tool-registration bookkeeping). Called once per incoming HTTP
// request so each request gets its own server<->transport pairing.
function createMcpServer(hrIndex, engIndex, adminIndex) {
  const server = new McpServer({ name: "hr-tools-server", version: "1.0.0" });

  // ---- 4 "live data" tools — unchanged logic from before ----

  server.registerTool(
    "get_leave_balance",
    {
      title: "Get Leave Balance",
      description: "Returns the employee's remaining leave balance for this year.",
      inputSchema: z.object({}),
    },
    async () => {
      const balance = randInt(20);
      console.log(`[SERVER] get_leave_balance called -> ${balance} days`);
      return { content: [{ type: "text", text: JSON.stringify({ remaining_leave_days: balance }) }] };
    }
  );

  server.registerTool(
    "get_my_projects",
    {
      title: "Get My Projects",
      description: "Returns the list of projects the employee is currently working on.",
      inputSchema: z.object({}),
    },
    async () => {
      const count = Math.max(1, randInt(10));
      const shuffled = [...PROJECT_NAMES].sort(() => 0.5 - Math.random());
      const projects = shuffled.slice(0, Math.min(count, PROJECT_NAMES.length));
      console.log(`[SERVER] get_my_projects called -> ${JSON.stringify(projects)}`);
      return { content: [{ type: "text", text: JSON.stringify({ projects }) }] };
    }
  );

  server.registerTool(
    "get_my_manager",
    {
      title: "Get My Manager",
      description: "Returns the name of the employee's manager.",
      inputSchema: z.object({}),
    },
    async () => {
      const manager = MANAGERS[randInt(MANAGERS.length - 1)];
      console.log(`[SERVER] get_my_manager called -> ${manager}`);
      return { content: [{ type: "text", text: manager }] };
    }
  );

  server.registerTool(
    "get_total_pto",
    {
      title: "Get Total PTO",
      description: "Returns the total number of PTO days allowed per year at the company.",
      inputSchema: z.object({}),
    },
    async () => {
      const pto = randInt(20);
      console.log(`[SERVER] get_total_pto called -> ${pto} days`);
      return { content: [{ type: "text", text: JSON.stringify({ total_pto_days: pto }) }] };
    }
  );

  // ---- 3 RAG tools — same retrieval logic as before, returning a
  // small JSON envelope (topScore + matches) instead of flat text, so
  // a caller can programmatically judge retrieval quality. ----

  server.registerTool(
    "search_hr_policy",
    {
      title: "Search HR Policy Docs",
      description:
        "Searches internal HR policy documents (leave, maternity/paternity, remote work, equipment, etc.) for a relevant answer. Use this for any HR policy question.",
      inputSchema: z.object({ query: z.string() }),
    },
    async ({ query }) => {
      console.log(`[SERVER] [RAG:hr-policy] search called -> "${query}"`);
      try {
        const matches = await hrIndex.search(query, 3);
        return ragToolResult("hr-policy", matches);
      } catch (err) {
        console.error(`[SERVER] [RAG:hr-policy] search FAILED for "${query}":`, err);
        throw err;
      }
    }
  );

  server.registerTool(
    "search_engineering_practices",
    {
      title: "Search Engineering Practice Docs",
      description:
        "Searches internal software engineering best-practice documents (Node.js, .NET, general coding/git/review practices) for a relevant answer. Use this for any coding standards or best-practice question.",
      inputSchema: z.object({ query: z.string() }),
    },
    async ({ query }) => {
      console.log(`[SERVER] [RAG:engineering] search called -> "${query}"`);
      try {
        const matches = await engIndex.search(query, 3);
        return ragToolResult("engineering", matches);
      } catch (err) {
        console.error(`[SERVER] [RAG:engineering] search FAILED for "${query}":`, err);
        throw err;
      }
    }
  );

  server.registerTool(
    "search_admin_docs",
    {
      title: "Search Admin / IT / Expense Policy Docs",
      description:
        "Searches internal admin documents (IT & security policy, expense/reimbursement policy) for a relevant answer. Use this for IT, security, or expense/reimbursement questions.",
      inputSchema: z.object({ query: z.string() }),
    },
    async ({ query }) => {
      console.log(`[SERVER] [RAG:admin-policies] search called -> "${query}"`);
      try {
        const matches = await adminIndex.search(query, 3);
        return ragToolResult("admin-policies", matches);
      } catch (err) {
        console.error(`[SERVER] [RAG:admin-policies] search FAILED for "${query}":`, err);
        throw err;
      }
    }
  );

  return server;
}

async function main() {
  console.log("[SERVER] Building RAG indices from docs/ ...");
  const hrIndex = await buildIndex(path.join(__dirname, "docs/hr-policy"), "hr-policy");
  const engIndex = await buildIndex(path.join(__dirname, "docs/engineering"), "engineering");
  const adminIndex = await buildIndex(path.join(__dirname, "docs/admin-policies"), "admin-policies");

  // ---- HTTP transport setup ----
  // Stateless mode (sessionIdGenerator: undefined). A single transport
  // instance can't safely be reused across multiple sequential
  // requests in this mode, so — same as before — we create a FRESH
  // transport per incoming request. What's new: we also create a
  // FRESH McpServer per request (via createMcpServer(), cheap — the
  // indices above are still only built once, at startup) instead of
  // reusing one shared server, since a shared server can only ever be
  // connected to one transport at a time.
  const app = express();
  app.use(express.json());

  app.post("/mcp", async (req, res) => {
    try {
      const server = createMcpServer(hrIndex, engIndex, adminIndex);
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
      res.on("close", () => {
        transport.close();
        server.close();
      });
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (err) {
      console.error("[SERVER] Error handling /mcp request:", err);
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: "2.0",
          error: { code: -32603, message: "Internal server error" },
          id: null,
        });
      }
    }
  });

  app.listen(PORT, () => {
    console.log(`[SERVER] HR MCP server running at http://localhost:${PORT}/mcp`);
    console.log("[SERVER] 7 tools registered. Waiting for client requests...");
  });
}

main().catch((err) => {
  console.error("[SERVER] Fatal error during startup:", err);
  process.exit(1);
});
