
"use client";

import { useState } from "react";
import { Button } from "@nextui-org/button";
import { Textarea } from "@nextui-org/input";

export interface RecommendationFormProps {
	onSubmit: (text: string) => void
}

export function RecommendationForm(props: RecommendationFormProps) {

    const [text, setText] = useState("");

    function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        props.onSubmit(text);
        event.preventDefault();
    }
    function handleTextAreaChange(event: React.FormEvent<HTMLFormElement>) {
        setText(text);
        event.preventDefault();
    }

    return (
        <form
            onSubmit={handleSubmit}
            className="flex flex-col items-start gap-4"
        >
            <div className="flex gap-3">
                <Textarea
                    label="Your preference:"
                    labelPlacement="outside"
                    placeholder="Describe the music that you would like to listen to"
                    className="max-w-xs"
                    value={text}
                    onChange={handleTextAreaChange}
                />
            </div>

            <div className="flex gap-3">
                <Button type="submit" color="primary" radius="full" variant="shadow">Get recommendations</Button>
            </div>
        </form>
    );
}