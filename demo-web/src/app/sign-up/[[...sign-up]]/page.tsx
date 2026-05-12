import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <main className="flex min-h-dvh items-center justify-center bg-vn-black px-6 py-20">
      <div className="flex w-full max-w-md flex-col items-center gap-6">
        <span className="vn-label text-vn-amber">Visual Narrator</span>
        <SignUp
          path="/sign-up"
          routing="path"
          signInUrl="/sign-in"
          forceRedirectUrl="/upload"
          appearance={{
            elements: {
              rootBox: "w-full",
              cardBox: "w-full shadow-none",
              card: "w-full border border-vn-line bg-transparent shadow-none",
              headerTitle: "hidden",
              headerSubtitle: "hidden",
              socialButtonsBlockButton:
                "border-vn-line bg-vn-carbon text-vn-cream shadow-none hover:bg-vn-ink",
              socialButtonsBlockButtonText: "text-vn-cream",
              dividerLine: "bg-vn-line",
              dividerText: "text-vn-dim",
              formFieldLabel: "vn-label text-vn-mist",
              formFieldInput:
                "border-vn-line bg-vn-carbon text-vn-cream placeholder:text-vn-dim focus:border-vn-amber",
              formButtonPrimary:
                "bg-vn-amber text-vn-black shadow-none hover:bg-amber-400",
              footerActionText: "text-vn-dim",
              footerActionLink: "text-vn-amber hover:text-amber-300",
              identityPreviewText: "text-vn-cream",
              identityPreviewEditButton: "text-vn-amber hover:text-amber-300",
              formResendCodeLink: "text-vn-amber hover:text-amber-300",
            },
          }}
        />
      </div>
    </main>
  );
}
