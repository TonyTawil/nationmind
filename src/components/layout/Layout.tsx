import { useState } from "react";
import { Sidebar } from "./Sidebar.tsx";
import { Header } from "./Header.tsx";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen bg-black text-white">
      <Sidebar collapsed={collapsed} />
      <div className="flex-1 flex flex-col">
        <Header collapsed={collapsed} setCollapsed={setCollapsed} />
        <main className="flex-1 p-6 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
