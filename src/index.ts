import express, { Request, Response, NextFunction } from "express";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { Pinecone } from "@pinecone-database/pinecone";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import dotenv from "dotenv";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnableBinding,
  RunnableLambda,
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import cors from "cors";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 9000;
const corsOptions = {
  origin: process.env.FRONTEND_URL, // React app URL
  optionsSuccessStatus: 200, // For legacy browser support
};

app.use(cors(corsOptions));

const template = `Answer the question based only on the following context:
{context}
Question: {question}`;

const prompt = ChatPromptTemplate.fromTemplate(template);

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
});

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

// Middleware to log requests
app.use((req: Request, res: Response, next: NextFunction) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

app.get("/api", (req: Request, res: Response) => {
  res.send({ message: "Hello from the server!" });
});

app.get(
  "/upsert",
  async (request: Request, response: Response, next: NextFunction) => {
    response.send("Chatbot processing started");
    // const loader = new JSONLoader("data.json");
    // const docs = await loader.load();
    // console.log("docs are, ", docs);
    const csvloader = new CSVLoader("data.csv");
    const csvdocs = await csvloader.load();
    console.log("csvdocs are, ", csvdocs);
    await PineconeStore.fromDocuments(
      csvdocs,
      new OpenAIEmbeddings({
        model: "text-embedding-3-small",
      }),
      {
        pineconeIndex,
        maxConcurrency: 5, // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
        namespace: "data1",
      }
    );
  }
);

app.get(
  "/getanswer",
  async (request: Request, response: Response, next: NextFunction) => {
    response.send("Chatbot processing started");
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings({
        model: "text-embedding-3-small",
      }),
      { pineconeIndex, namespace: "data1" }
    );
    /* Search the vector DB independently with metadata filters */
    // const results = await vectorStore.similaritySearch("data1");
    const chain = RunnableSequence.from([
      RunnablePassthrough.assign({
        context: async (input: any, config) => {
          if (!config || !("configurable" in config)) {
            throw new Error("No config");
          }
          const { configurable } = config;
          const documents = await vectorStore
            .asRetriever(configurable)
            .invoke(input.question, config);
          return documents.map((doc) => doc.pageContent).join("\n\n");
        },
      }),
      prompt,
      model,
      new StringOutputParser(),
    ]);
    let finalAnswer = await chain.invoke(
      { question: "Show me the all cars you have with all details?" },
      { configurable: { filter: { namespace: "data1" } } }
    );
    console.log("finalAnswer is, ", finalAnswer);
  }
);

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).send("Something broke!");
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
