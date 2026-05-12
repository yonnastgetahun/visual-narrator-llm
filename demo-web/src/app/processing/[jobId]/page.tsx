import { ProcessingScreen } from "@/components/processing/ProcessingScreen";

export const metadata = {
  title: "Processing — Visual Narrator",
};

export default function ProcessingPage({ params }: { params: { jobId: string } }) {
  return <ProcessingScreen jobId={params.jobId} />;
}
