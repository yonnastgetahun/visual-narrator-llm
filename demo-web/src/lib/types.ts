export type StepEvent =
  | { step: "download" | "gaps"; message: string }
  | { step: "gaps_done"; gaps: number; duration_seconds: number }
  | { step: "describing" | "synthesizing" | "scoring"; current: number; total: number; message: string };

export type NarrationEntry = {
  srt_index: number;
  start_sec: number;
  end_sec: number;
  gap_duration_sec: number;
  gap_type: string;
  frame_timestamp_sec: number;
  description: string;
  audio_data: string;
  audio_mime: string;
  gpt_cost: number;
  tts_cost: number;
};

export type Manifest = {
  job_id?: string;
  source: string;
  duration_seconds: number;
  gaps_found: number;
  model_version: string;
  voice_id: string;
  compliance_level: string;
  gpt_cost_estimate: number;
  tts_cost_estimate: number;
  total_cost_estimate: number;
  compliance: Record<string, unknown>;
  narrations: NarrationEntry[];
};

export type ScoreEntry = {
  srt_index: number;
  start_sec: number;
  end_sec: number;
  frame_timestamp_sec: number;
  description: string;
  accuracy: number;
  relevance: number;
  wcag_compliance: number;
  conciseness: number;
  overall: number;
  word_count: number;
  tense_ok: boolean;
  within_limit: boolean;
  flag: boolean;
  flag_reason: string | null;
  gpt_cost: number;
};

export type ScoreReport = {
  source: string;
  manifest: string;
  scored: number;
  flagged: number;
  word_limit: number;
  aggregate: {
    accuracy: number;
    relevance: number;
    wcag_compliance: number;
    conciseness: number;
    overall: number;
    within_limit_pct: number;
    tense_ok_pct: number;
  };
  grade: string;
  gpt_cost_estimate: number;
  scores: ScoreEntry[];
};
