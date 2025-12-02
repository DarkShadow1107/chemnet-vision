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
	images: {
		remotePatterns: [
			{
				protocol: "http",
				hostname: "localhost",
				port: "5000",
				pathname: "/images/**",
			},
			{
				protocol: "http",
				hostname: "127.0.0.1",
				port: "5000",
				pathname: "/images/**",
			},
		],
	},
};

export default nextConfig;
