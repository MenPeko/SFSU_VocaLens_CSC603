import {VQA_PROMPT} from "../../src/services/gemini-vqa"

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

export function compareModelName(env: NodeJS.ProcessEnv = process.env): string {
  const model = env.VOCALENS_COMPARE_MODEL?.trim() || ""
  if (!model) {
    throw new Error("Set VOCALENS_COMPARE_MODEL to an OpenAI vision model. The live app keeps using Gemini. Nothing was compared.")
  }
  if (model.toLowerCase().includes("gemini")) {
    throw new Error("VOCALENS_COMPARE_MODEL must name the other hosted vision model. Gemini is already one side of the comparison.")
  }
  return model
}

export function buildCompareVisionBody(
  model: string,
  prompt: string,
  question: string,
  imageBase64: string,
  mimeType: string,
): string {
  return JSON.stringify({
    model,
    messages: [
      {
        role: "user",
        content: [
          {type: "text", text: `${prompt}\n\nQuestion: ${question}`},
          {
            type: "image_url",
            image_url: {url: `data:${mimeType || "image/jpeg"};base64,${imageBase64}`},
          },
        ],
      },
    ],
  })
}

export function textFromOpenAiVisionResponse(payload: {
  choices?: Array<{message?: {content?: string | Array<{text?: string}> | null}}>
}): string {
  const content = payload.choices?.[0]?.message?.content
  if (typeof content === "string") return content.trim()
  if (Array.isArray(content)) {
    return content
      .map((part) => part.text || "")
      .join("")
      .trim()
  }
  return ""
}

export async function answerWithCompareModel(image: Buffer, mimeType: string, question: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY?.trim() || ""
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set. Nothing was compared.")
  }

  const model = compareModelName()
  const response = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: buildCompareVisionBody(model, VQA_PROMPT, question, image.toString("base64"), mimeType),
  })

  if (!response.ok) {
    throw new Error(`Comparison vision request failed: ${response.status} ${response.statusText}`)
  }

  return textFromOpenAiVisionResponse(await response.json())
}
