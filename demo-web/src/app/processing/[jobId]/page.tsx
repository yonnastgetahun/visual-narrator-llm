import { ProcessingScreen } from "@/components/processing/ProcessingScreen";

export const metadata = {
  title: "Processing — Visual Narrator",
};

export default function ProcessingPage({
  params,
  searchParams,
}: {
  params: { jobId: string };
  searchParams?: { minutes?: string };
}) {
  const estimatedMinutes = Number(searchParams?.minutes ?? "90");

  return <ProcessingScreen estimatedMinutes={estimatedMinutes} jobId={params.jobId} />;
}
