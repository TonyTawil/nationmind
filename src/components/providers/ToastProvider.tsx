import { Toaster } from "sonner";

export function ToastProvider() {
  return (
    <Toaster
      position="top-right"
      toastOptions={{
        style: {
          background: "#1a1a1a",
          color: "#ffffff",
          border: "1px solid #333",
        },
        duration: 3000,
      }}
    />
  );
}
