import { ResultsScreen } from "@/components/results/ResultsScreen";

export const metadata = {
  title: "Your AD Track is Ready — Visual Narrator",
};

export default function ResultsPage({
  params,
  searchParams,
}: {
  params: { jobId: string };
  searchParams?: {
    duration?: string;
    gaps?: string;
    gpt?: string;
    total?: string;
    tts?: string;
    wcag?: string;
  };
}) {
  return <ResultsScreen jobId={params.jobId} summary={searchParams} />;
}
