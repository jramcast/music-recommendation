const express = require('express');
const { recommend } = require('../recommendation/Pipeline');
const { UserPreferenceAsText } = require('../recommendation/entities');
// eslint-disable-next-line new-cap
const router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', {title: 'Music Recommendation'});
});

router.post('/recommend', async function(req, res, next) {
    const preference = new UserPreferenceAsText(req.body.preference);
    const recommendation = await recommend(preference);
    res.render('recommendation', { preference, recommendation});
});

export default router;
