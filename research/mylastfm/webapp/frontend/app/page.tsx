"use client";
import { Link } from "@nextui-org/link";
import { Button } from "@nextui-org/button";

import { Snippet } from "@nextui-org/snippet";
import { Code } from "@nextui-org/code"
import { button as buttonStyles } from "@nextui-org/theme";
import { siteConfig } from "@/config/site";
import { title, subtitle } from "@/components/primitives";
import { GithubIcon } from "@/components/icons";
import { Textarea, Input } from "@nextui-org/input";

export default function Home() {


	function handleSubmit(event) {
		console.log(event);
		event.preventDefault();
	}

	return (
		<section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
			<div className="inline-block max-w-lg text-center justify-center">
				<h1 className={title()}>Get&nbsp;</h1>
				<h1 className={title({ color: "violet" })}>music recommendations&nbsp;</h1>
				<br />
				<h1 className={title()}>
					based on your preferences.
				</h1>
				<h2 className={subtitle({ class: "mt-4" })}>
					Explainable recommendations
				</h2>
			</div>



			{/* <div className="mt-8">
				<Snippet hideSymbol hideCopyButton variant="flat">
					<span>
						Get started by editing <Code color="primary">app/page.tsx</Code>
					</span>
				</Snippet>
			</div> */}
		</section>
	);
}
