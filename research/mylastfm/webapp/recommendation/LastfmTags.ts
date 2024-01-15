import { LastfmTag } from "./entities";
import { Track } from "./entities";
import { MongoClient } from "mongodb";

const MONGODB_ADDRESS = process.env.MONGODB_ADDRESS;

export async function findLastfmTagsForTrack(track: Track) {

    const client = new MongoClient(MONGODB_ADDRESS);

    try {
        await client.connect();
        const db = client.db("mgr");

        const lastfmPlayedtracks = db.collection("lastfm_playedtracks");
        const result = await lastfmPlayedtracks.findOne(
                {
                    "artist.name": track.artist,
                    "name": track.trackName
                },
                {
                    projection: {
                        tags: true,
                        artist_tags: true,
                    }
                }
            );

        let tags;

        if (Array.isArray(result.tags) && result.tags.length > 0) {
            tags = result.tags;
        } else if (Array.isArray(result.rtist_tags) && result.artist_tags.length > 0) {
            tags =  result.artist_tags;
        } else {
            tags = [];
        }

        return tags.map(asLastFmTag);

    } finally {
        await client.close();
    }

}

function asLastFmTag(tagFromDatabase): LastfmTag  {
    return {
        tag: tagFromDatabase[0],
        weight: tagFromDatabase[1]
    }
}