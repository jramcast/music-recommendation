import { countSpotifyTracks } from "../search/spotify";

const express = require('express');
const { recommend } = require('../recommendation/Pipeline');
const { UserPreferenceAsText } = require('../recommendation/entities');
const { findSpotifyTracks } = require('../search/spotify');
import { fetchRecommendations } from "../recommendation/TracksRanking";
import { Recommendation } from "../recommendation/entities";


// eslint-disable-next-line new-cap
const router = express.Router();

/* GET home page. */
router.get('/', async function(req, res, next) {
  const page = parseInt(req.query.page) || 1;
  const perPage = 10;
  const searchTerm = req.query.q || "";
  const songs = await findSpotifyTracks(searchTerm, page);
  const totalSongs = await countSpotifyTracks(searchTerm);

  res.render('index', {
    title: 'Music Recommendation',
    songs,
    searchTerm,
    currentPage: page,
    totalPages: Math.ceil(totalSongs / perPage)
  });
});


router.post('/recommend', async (req, res, next) => {
    const preference = new UserPreferenceAsText(req.body.preference);
    const recommendation = await recommend(preference);
    res.render('recommendation', { preference, recommendation});
});

router.get('/recommend', async (req, res, next) => {
    const audioFeatures = req.query;
    let tracks = await fetchRecommendations(audioFeatures);
    const recommendation = new Recommendation(tracks, audioFeatures);
    res.render('recommendation', { preference: {text: JSON.stringify(audioFeatures)}, recommendation});
});

export default router;