/**
 * Anthropic media-understanding provider descriptor. It routes image and native
 * document description through the shared model-backed media helpers.
 */
import {
  type JsonSchemaObject,
  validateJsonSchemaValue,
} from "openclaw/plugin-sdk/json-schema-runtime";
import {
  describeImageWithModel,
  describeImagesWithModel,
  type MediaUnderstandingProvider,
  type StructuredExtractionImageInput,
  type StructuredExtractionRequest,
  type StructuredExtractionResult,
  type StructuredExtractionTextInput,
} from "openclaw/plugin-sdk/media-understanding";

/** Media-understanding provider for Anthropic Claude models. */
export const anthropicMediaUnderstandingProvider: MediaUnderstandingProvider = {
  id: "anthropic",
  capabilities: ["image"],
  defaultModels: { image: "claude-opus-5" },
  autoPriority: { image: 20 },
  nativeDocumentInputs: ["pdf"],
  describeImage: describeImageWithModel,
  describeImages: describeImagesWithModel,
  extractStructured: extractAnthropicStructured,
};

function isStructuredImageInput(
  entry: StructuredExtractionRequest["input"][number],
): entry is StructuredExtractionImageInput {
  return entry.type === "image";
}

function isStructuredTextInput(
  entry: StructuredExtractionRequest["input"][number],
): entry is StructuredExtractionTextInput {
  return entry.type === "text";
}

async function extractAnthropicStructured(
  req: StructuredExtractionRequest,
): Promise<StructuredExtractionResult> {
  const model = req.model.trim();
  if (!model) {
    throw new Error("Anthropic structured extraction requires model id.");
  }
  const instructions = req.instructions.trim();
  if (!instructions) {
    throw new Error("Anthropic structured extraction requires instructions.");
  }
  if (req.input.length === 0) {
    throw new Error("Anthropic structured extraction requires at least one input.");
  }
  const images = req.input.filter(isStructuredImageInput);
  if (images.length === 0) {
    throw new Error("Anthropic structured extraction requires at least one image input.");
  }
  req.signal?.throwIfAborted();

  const { text } = await describeImagesWithModel({
    images: images.map((image) => ({
      buffer: image.buffer,
      fileName: image.fileName,
      mime: image.mime,
    })),
    model,
    provider: req.provider,
    prompt: buildAnthropicStructuredExtractionPrompt(req),
    timeoutMs: req.timeoutMs,
    ...(req.signal ? { signal: req.signal } : {}),
    profile: req.profile,
    preferredProfile: req.preferredProfile,
    authStore: req.authStore,
    agentDir: req.agentDir,
    cfg: req.cfg,
  });

  return normalizeAnthropicStructuredExtractionResult({
    text,
    model,
    provider: req.provider,
    req,
  });
}

function buildAnthropicStructuredExtractionPrompt(req: StructuredExtractionRequest): string {
  const textInputs = req.input.filter(isStructuredTextInput).map((entry) => entry.text);
  return [
    req.instructions.trim(),
    req.schemaName ? `Schema name: ${req.schemaName}` : undefined,
    req.jsonSchema ? `JSON schema:\n${JSON.stringify(req.jsonSchema)}` : undefined,
    ...textInputs,
    req.jsonMode === false
      ? "Return the extraction as concise text."
      : "Return valid JSON only. Do not wrap the JSON in Markdown fences.",
  ]
    .filter((part): part is string => Boolean(part))
    .join("\n\n");
}

function isJsonSchemaObject(value: unknown): value is JsonSchemaObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeAnthropicStructuredExtractionResult(params: {
  text: string;
  model: string;
  provider: string;
  req: StructuredExtractionRequest;
}): StructuredExtractionResult {
  const result: StructuredExtractionResult = {
    text: params.text,
    model: params.model,
    provider: params.provider,
    contentType: params.req.jsonMode === false ? "text" : "json",
  };
  if (params.req.jsonMode !== false) {
    try {
      result.parsed = JSON.parse(params.text);
    } catch {
      throw new Error("Anthropic structured extraction returned invalid JSON.");
    }
    if (isJsonSchemaObject(params.req.jsonSchema)) {
      const validation = validateJsonSchemaValue({
        schema: params.req.jsonSchema,
        cacheKey: "anthropic.media-understanding.extractStructured",
        value: result.parsed,
        cache: false,
      });
      if (!validation.ok) {
        const message = validation.errors.map((error) => error.text).join("; ") || "invalid";
        throw new Error(`Anthropic structured extraction JSON did not match schema: ${message}`);
      }
      result.parsed = validation.value;
    }
  }
  return result;
}
