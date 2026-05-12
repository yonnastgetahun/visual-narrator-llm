import { ResultsScreen } from "@/components/results/ResultsScreen";

export const metadata = {
  title: "Your AD Track is Ready — Visual Narrator",
};

export default function ResultsPage({ params }: { params: { jobId: string } }) {
  return <ResultsScreen jobId={params.jobId} />;
}
