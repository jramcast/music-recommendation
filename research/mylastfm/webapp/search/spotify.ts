import { Track } from "../recommendation/entities";

const { MongoClient, ServerApiVersion } = require('mongodb');

const MONGODB_ADDRESS = process.env.MONGODB_ADDRESS;

export async function countSpotifyTracks(searchTerm: string) {
    const client = new MongoClient(MONGODB_ADDRESS);
    try {
        const spotifyAudiofeatures = await connect(client);

        const query = createQueryFromSearchTerm(searchTerm);

        // Count the total number of songs matching the search term
        return await spotifyAudiofeatures.countDocuments(query);


    } finally {
        await client.close();
    }
}

export async function findSpotifyTracks(searchTerm: string, page = 1, perPage = 10) {
    const client = new MongoClient(MONGODB_ADDRESS);

    try {
        const spotifyAudiofeatures = await connect(client);

        const query = createQueryFromSearchTerm(searchTerm);

        const songs = await spotifyAudiofeatures.find(query)
            .project({
                features: true,
                track_name: true,
                track_artist: true
            })
            .skip((page - 1) * perPage)
            .limit(perPage)
            .toArray();


        return songs.map(doc => new Track(doc.track_artist, doc.track_name, doc.features[0]));

    } finally {
        await client.close();
    }

}

async function connect(client: any) {
    await client.connect();
    const db = client.db("mgr");
    const spotifyAudiofeatures = db.collection("spotify_audiofeatures");
    return spotifyAudiofeatures;
}



function createQueryFromSearchTerm(searchTerm: string) {
    // Construct a regex pattern for advanced search
    const regexPattern = escapeRegExp(searchTerm);//;.split(' ').join('|');

    // Query MongoDB for songs matching the search term in title or artist, with pagination
    return {
        $or: [
            { track_name: { $regex: regexPattern, $options: 'i' } },
            { track_artist: { $regex: regexPattern, $options: 'i' } }
        ]
    };
}

// Function to escape special characters in a string for regex
function escapeRegExp(text: string) {
    return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
}