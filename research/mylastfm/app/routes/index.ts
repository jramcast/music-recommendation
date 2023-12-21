const express = require('express');
const { recommend } = require('../recommendation/Pipeline');
const { TextPreference } = require('../recommendation/entities');
// eslint-disable-next-line new-cap
const router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', {title: 'Music Recommendation'});
});

router.post('/recommend', async function(req, res, next) {
    const preference = new TextPreference(req.body.preference);
    const recommendation = await recommend(preference);
    console.log(recommendation);
    res.render('recommendation', { preference, recommendation});
});

export default router;