import { Moon, LayoutDashboard, ChevronDown } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useGraph } from "@/contexts/GraphContext";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface HeaderProps {
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

export function Header({ collapsed, setCollapsed }: HeaderProps) {
  const location = useLocation();
  const currentPath = location.pathname;
  const { currentGraph, graphs, setCurrentGraphById } = useGraph();

  // Function to get the current page name from the path
  const getCurrentPageName = () => {
    switch (currentPath) {
      case "/":
        return "Home";
      case "/knowledge-explorer":
        return "Knowledge Explorer";
      case "/chat":
        return "Chat";
      case "/data":
        return "Data";
      case "/configuration":
        return "Configuration";
      case "/docs":
        return "Docs";
      default:
        return "Home";
    }
  };

  return (
    <header className="h-14 border-b border-gray-800 flex items-center justify-between px-4">
      <div className="flex items-center">
        <Button
          variant="ghost"
          size="icon"
          className="mr-4"
          onClick={() => setCollapsed(!collapsed)}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M1.5 3C1.22386 3 1 3.22386 1 3.5C1 3.77614 1.22386 4 1.5 4H13.5C13.7761 4 14 3.77614 14 3.5C14 3.22386 13.7761 3 13.5 3H1.5ZM1 7.5C1 7.22386 1.22386 7 1.5 7H13.5C13.7761 7 14 7.22386 14 7.5C14 7.77614 13.7761 8 13.5 8H1.5C1.22386 8 1 7.77614 1 7.5ZM1 11.5C1 11.2239 1.22386 11 1.5 11H13.5C13.7761 11 14 11.2239 14 11.5C14 11.7761 13.7761 12 13.5 12H1.5C1.22386 12 1 11.7761 1 11.5Z"
              fill="currentColor"
              fillRule="evenodd"
              clipRule="evenodd"
            ></path>
          </svg>
        </Button>
        <div className="flex items-center">
          <Button
            variant="link"
            className="p-0 h-auto flex items-center text-foreground"
            asChild
          >
            <Link to="/">
              <LayoutDashboard size={16} className="mr-2" />
              <span>Dashboard</span>
            </Link>
          </Button>
        </div>
        <Separator orientation="vertical" className="mx-2 h-4" />
        <Button variant="link" className="p-0 h-auto text-foreground" asChild>
          <Link to={currentPath}>{getCurrentPageName()}</Link>
        </Button>
      </div>
      <div className="flex items-center gap-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" className="flex items-center gap-2">
              <div className="h-4 w-4 rounded-full bg-white flex items-center justify-center">
                <div className="h-2 w-2 rounded-full bg-black"></div>
              </div>
              <span>{currentGraph?.name || "No graph selected"}</span>
              <ChevronDown size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {graphs.length === 0 ? (
              <DropdownMenuItem disabled>No graphs available</DropdownMenuItem>
            ) : (
              graphs.map((graph) => (
                <DropdownMenuItem
                  key={graph.id}
                  onClick={() => setCurrentGraphById(graph.id)}
                  className={currentGraph?.id === graph.id ? "bg-gray-800" : ""}
                >
                  {graph.name}
                </DropdownMenuItem>
              ))
            )}
            <Separator className="my-1" />
            <DropdownMenuItem asChild>
              <Link to="/configuration" className="cursor-pointer">
                Create new graph
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant="ghost" size="icon" className="rounded-full">
          <Moon size={20} />
        </Button>
      </div>
    </header>
  );
}
