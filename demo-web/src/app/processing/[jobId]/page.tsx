import { ProcessingScreen } from "@/components/processing/ProcessingScreen";

export const metadata = {
  title: "Processing — Visual Narrator",
};

export default function ProcessingPage({
  params,
  searchParams,
}: {
  params: { jobId: string };
  searchParams?: { minutes?: string; s3Key?: string; source?: string };
}) {
  const estimatedMinutes = Number(searchParams?.minutes ?? "90");
  const s3Key = searchParams?.s3Key ?? undefined;
  const source = searchParams?.source ?? undefined;

  return (
    <ProcessingScreen
      estimatedMinutes={estimatedMinutes}
      jobId={params.jobId}
      s3Key={s3Key}
      source={source}
    />
  );
}
