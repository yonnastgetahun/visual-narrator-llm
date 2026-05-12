import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { auth } from "@clerk/nextjs/server";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { NextRequest, NextResponse } from "next/server";

const UPLOAD_BUCKET = process.env.S3_UPLOAD_BUCKET ?? "vnpoverview-uploads";
const UPLOAD_REGION = process.env.AWS_REGION ?? "us-east-1";
const PRESIGN_EXPIRY_SECONDS = 60 * 60;

const s3Client = new S3Client({ region: UPLOAD_REGION });

export const runtime = "nodejs";

function sanitizeFilename(filename: string): string {
  const basename = filename.split(/[/\\]/).pop()?.trim() || "upload.mp4";
  const sanitized = basename
    .replace(/[^\w.()\- ]+/g, "-")
    .replace(/\s+/g, "-")
    .slice(0, 200);

  return sanitized || "upload.mp4";
}

export async function POST(request: NextRequest) {
  const { userId } = await auth();
  if (!userId) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const body = (await request.json().catch(() => null)) as
    | { filename?: unknown; contentType?: unknown }
    | null;
  const filename = typeof body?.filename === "string" ? body.filename : "";
  const contentType =
    typeof body?.contentType === "string" && body.contentType.trim()
      ? body.contentType
      : "application/octet-stream";

  if (!filename.trim()) {
    return NextResponse.json({ error: "Filename is required." }, { status: 400 });
  }

  const s3Key = `uploads/${userId}/${Date.now()}-${sanitizeFilename(filename)}`;
  const command = new PutObjectCommand({
    Bucket: UPLOAD_BUCKET,
    Key: s3Key,
    ContentType: contentType,
  });
  const url = await getSignedUrl(s3Client, command, {
    expiresIn: PRESIGN_EXPIRY_SECONDS,
  });

  return NextResponse.json({
    url,
    s3_key: s3Key,
    expires_in: PRESIGN_EXPIRY_SECONDS,
  });
}
