import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
	/* config options here */
	reactCompiler: true,
	// Ensure Turbopack uses this project directory (local package-lock.json)
	// This forces Next.js to treat this folder as the workspace root.
	turbopack: {
		root: path.resolve(__dirname),
	} as any,
};

export default nextConfig;
