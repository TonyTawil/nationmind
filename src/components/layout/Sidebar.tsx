import {
  Home,
  Database,
  MessageSquare,
  Settings,
  BookOpen,
  Github,
  MessagesSquare,
  Mail,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

interface SidebarProps {
  collapsed: boolean;
}

export function Sidebar({ collapsed }: SidebarProps) {
  const location = useLocation();
  const currentPath = location.pathname;

  // Common button styles for all navigation items
  const buttonClasses = (isActive: boolean) =>
    cn(
      "flex items-center w-full h-10 rounded-md transition-colors overflow-hidden",
      isActive
        ? "bg-gray-800 text-white hover:bg-gray-700"
        : "text-gray-400 hover:text-white hover:bg-gray-800/50"
    );

  // Common icon container styles
  const iconContainerClasses =
    "flex items-center justify-center w-10 h-10 flex-shrink-0";

  // Text label styles with transition
  const textClasses = cn(
    "whitespace-nowrap transition-all duration-300 ease-in-out",
    collapsed ? "w-0 opacity-0" : "w-auto opacity-100 ml-1"
  );

  return (
    <div
      className={cn(
        "bg-[#0f0f13] border-r border-gray-800 flex flex-col transition-all duration-300 ease-in-out",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Logo - adjust height to match header */}
      <div className="h-14 flex items-center px-4 border-b border-gray-800 overflow-hidden">
        <div className="h-8 w-8 rounded-full bg-white flex items-center justify-center flex-shrink-0">
          <div className="h-4 w-4 rounded-full bg-black"></div>
        </div>
        <span
          className={cn(
            "ml-3 font-bold text-lg whitespace-nowrap transition-all duration-300 ease-in-out",
            collapsed ? "w-0 opacity-0" : "w-auto opacity-100"
          )}
        >
          Circlemind
        </span>
      </div>

      {/* External links */}
      <div className="p-2 flex flex-col gap-1">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className={buttonClasses(false)}
        >
          <div className={iconContainerClasses}>
            <Github size={20} />
          </div>
          <span className={textClasses}>GitHub</span>
        </a>

        <a
          href="https://discord.com"
          target="_blank"
          rel="noopener noreferrer"
          className={buttonClasses(false)}
        >
          <div className={iconContainerClasses}>
            <MessagesSquare size={20} />
          </div>
          <span className={textClasses}>Discord</span>
        </a>

        <a href="mailto:support@example.com" className={buttonClasses(false)}>
          <div className={iconContainerClasses}>
            <Mail size={20} />
          </div>
          <span className={textClasses}>Support</span>
        </a>
      </div>

      <Separator className="my-2 mx-auto w-[90%]" />

      {/* Main navigation */}
      <div className="flex-1 p-2 flex flex-col gap-1">
        <Link to="/" className={buttonClasses(currentPath === "/")}>
          <div className={iconContainerClasses}>
            <Home size={20} />
          </div>
          <span className={textClasses}>Home</span>
        </Link>

        <Link
          to="/knowledge-explorer"
          className={buttonClasses(currentPath === "/knowledge-explorer")}
        >
          <div className={iconContainerClasses}>
            <BookOpen size={20} />
          </div>
          <span className={textClasses}>Knowledge Explorer</span>
        </Link>

        <Link to="/chat" className={buttonClasses(currentPath === "/chat")}>
          <div className={iconContainerClasses}>
            <MessageSquare size={20} />
          </div>
          <span className={textClasses}>Chat</span>
        </Link>

        <Link to="/data" className={buttonClasses(currentPath === "/data")}>
          <div className={iconContainerClasses}>
            <Database size={20} />
          </div>
          <span className={textClasses}>Data</span>
        </Link>

        <Link
          to="/configuration"
          className={buttonClasses(currentPath === "/configuration")}
        >
          <div className={iconContainerClasses}>
            <Settings size={20} />
          </div>
          <span className={textClasses}>Configuration</span>
        </Link>

        <Link to="/docs" className={buttonClasses(false)}>
          <div className={iconContainerClasses}>
            <BookOpen size={20} />
          </div>
          <span className={textClasses}>Docs</span>
        </Link>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800 flex items-center justify-between">
        <div className="flex items-center">
          <div
            className={cn(
              "text-xs text-gray-400 transition-all duration-300 ease-in-out",
              collapsed ? "w-0 opacity-0 overflow-hidden" : "w-auto opacity-100"
            )}
          >
            <div>2 / 100 requests</div>
            <div>Plan: Free</div>
          </div>
        </div>
        <div
          className={cn(
            "transition-all duration-300 ease-in-out",
            collapsed ? "w-0 opacity-0 overflow-hidden" : "w-auto opacity-100"
          )}
        >
          <Button variant="outline" size="sm" className="text-xs">
            Upgrade
          </Button>
        </div>
      </div>
    </div>
  );
}
