// Anthropic tests cover media understanding provider plugin behavior.
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  describeImagesWithModel: vi.fn(),
}));

vi.mock("openclaw/plugin-sdk/media-understanding", async (importOriginal) => ({
  ...(await importOriginal<typeof import("openclaw/plugin-sdk/media-understanding")>()),
  describeImagesWithModel: mocks.describeImagesWithModel,
}));

import { anthropicMediaUnderstandingProvider } from "./media-understanding-provider.js";

function baseImageInput() {
  return {
    type: "image" as const,
    buffer: Buffer.from("image-bytes"),
    fileName: "image.png",
    mime: "image/png",
  };
}

beforeEach(() => {
  mocks.describeImagesWithModel.mockReset();
});

describe("anthropicMediaUnderstandingProvider", () => {
  it("has expected provider metadata", () => {
    expect(anthropicMediaUnderstandingProvider.id).toBe("anthropic");
    expect(anthropicMediaUnderstandingProvider.capabilities).toEqual(["image"]);
    expect(anthropicMediaUnderstandingProvider.nativeDocumentInputs).toEqual(["pdf"]);
    expect(typeof anthropicMediaUnderstandingProvider.describeImage).toBe("function");
    expect(typeof anthropicMediaUnderstandingProvider.describeImages).toBe("function");
    expect(typeof anthropicMediaUnderstandingProvider.extractStructured).toBe("function");
  });

  it("routes structured extraction through describeImagesWithModel and parses JSON", async () => {
    mocks.describeImagesWithModel.mockResolvedValueOnce({
      text: '{"summary":"ok"}',
      model: "claude-sonnet-5",
    });

    const result = await anthropicMediaUnderstandingProvider.extractStructured?.({
      input: [{ type: "text", text: "Extract searchable evidence." }, baseImageInput()],
      instructions: "Return summary JSON.",
      provider: "anthropic",
      model: "claude-sonnet-5",
      timeoutMs: 30_000,
      cfg: {},
      agentDir: "/tmp/openclaw-agent",
    });

    expect(mocks.describeImagesWithModel).toHaveBeenCalledTimes(1);
    const call = mocks.describeImagesWithModel.mock.calls[0]?.[0];
    expect(call.model).toBe("claude-sonnet-5");
    expect(call.provider).toBe("anthropic");
    expect(call.images).toEqual([
      { buffer: Buffer.from("image-bytes"), fileName: "image.png", mime: "image/png" },
    ]);
    expect(call.prompt).toContain("Return summary JSON.");
    expect(call.prompt).toContain("Extract searchable evidence.");
    expect(call.prompt).toContain("Return valid JSON only");

    expect(result).toEqual({
      text: '{"summary":"ok"}',
      model: "claude-sonnet-5",
      provider: "anthropic",
      contentType: "json",
      parsed: { summary: "ok" },
    });
  });

  it("returns text content without parsing when jsonMode is false", async () => {
    mocks.describeImagesWithModel.mockResolvedValueOnce({
      text: "a plain description",
      model: "claude-sonnet-5",
    });

    const result = await anthropicMediaUnderstandingProvider.extractStructured?.({
      input: [baseImageInput()],
      instructions: "Describe the image.",
      jsonMode: false,
      provider: "anthropic",
      model: "claude-sonnet-5",
      timeoutMs: 30_000,
      cfg: {},
      agentDir: "/tmp/openclaw-agent",
    });

    const call = mocks.describeImagesWithModel.mock.calls[0]?.[0];
    expect(call.prompt).toContain("Return the extraction as concise text.");
    expect(result).toEqual({
      text: "a plain description",
      model: "claude-sonnet-5",
      provider: "anthropic",
      contentType: "text",
    });
  });

  it("rejects text-only structured extraction before calling the model", async () => {
    await expect(
      anthropicMediaUnderstandingProvider.extractStructured?.({
        input: [{ type: "text", text: "The answer is only text." }],
        instructions: "Return summary JSON.",
        provider: "anthropic",
        model: "claude-sonnet-5",
        timeoutMs: 30_000,
        cfg: {},
        agentDir: "/tmp/openclaw-agent",
      }),
    ).rejects.toThrow("Anthropic structured extraction requires at least one image input.");
    expect(mocks.describeImagesWithModel).not.toHaveBeenCalled();
  });

  it("rejects a missing model id before calling the model", async () => {
    await expect(
      anthropicMediaUnderstandingProvider.extractStructured?.({
        input: [baseImageInput()],
        instructions: "Return summary JSON.",
        provider: "anthropic",
        model: "  ",
        timeoutMs: 30_000,
        cfg: {},
        agentDir: "/tmp/openclaw-agent",
      }),
    ).rejects.toThrow("Anthropic structured extraction requires model id.");
    expect(mocks.describeImagesWithModel).not.toHaveBeenCalled();
  });

  it("rejects missing instructions before calling the model", async () => {
    await expect(
      anthropicMediaUnderstandingProvider.extractStructured?.({
        input: [baseImageInput()],
        instructions: "   ",
        provider: "anthropic",
        model: "claude-sonnet-5",
        timeoutMs: 30_000,
        cfg: {},
        agentDir: "/tmp/openclaw-agent",
      }),
    ).rejects.toThrow("Anthropic structured extraction requires instructions.");
    expect(mocks.describeImagesWithModel).not.toHaveBeenCalled();
  });

  it("returns a controlled error when structured JSON parsing fails", async () => {
    mocks.describeImagesWithModel.mockResolvedValueOnce({
      text: "not json",
      model: "claude-sonnet-5",
    });

    await expect(
      anthropicMediaUnderstandingProvider.extractStructured?.({
        input: [{ type: "text", text: "Extract JSON." }, baseImageInput()],
        instructions: "Return summary JSON.",
        provider: "anthropic",
        model: "claude-sonnet-5",
        timeoutMs: 30_000,
        cfg: {},
        agentDir: "/tmp/openclaw-agent",
      }),
    ).rejects.toThrow("Anthropic structured extraction returned invalid JSON.");
  });

  it("validates structured extraction JSON against the requested schema", async () => {
    mocks.describeImagesWithModel.mockResolvedValueOnce({
      text: '{"summary":123,"tags":["shape"]}',
      model: "claude-sonnet-5",
    });

    await expect(
      anthropicMediaUnderstandingProvider.extractStructured?.({
        input: [{ type: "text", text: "Extract JSON." }, baseImageInput()],
        instructions: "Return summary JSON.",
        jsonSchema: {
          type: "object",
          properties: {
            summary: { type: "string" },
          },
          required: ["summary"],
        },
        provider: "anthropic",
        model: "claude-sonnet-5",
        timeoutMs: 30_000,
        cfg: {},
        agentDir: "/tmp/openclaw-agent",
      }),
    ).rejects.toThrow("Anthropic structured extraction JSON did not match schema");
  });
});
