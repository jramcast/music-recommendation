envexport:
	conda env export --from-history | grep -v "^prefix: " > environment.yml

mongo:
	podman start mongomgr

mongo_create:
	podman run --name mongomgr -d -v mongomgr:/data/db/ -v $$(pwd)/data:/tmp/mgr-export:z  -p 27017:27017 mongo:4

mongocli:
	podman exec -i -t mongomgr mongo mgr

mongoexport:
	podman exec -i -t mongomgr mongoexport --uri=mongodb://localhost:27017/mgr --collection lastfm_playedtracks --out /tmp/mgr-export/lastfm_playedtracks.json
	podman exec -i -t mongomgr mongoexport --uri=mongodb://localhost:27017/mgr --collection spotify_audiofeatures --out /tmp/mgr-export/spotify_audiofeatures.json

mongoimport:
	podman exec -i -t mongomgr mongoimport --uri=mongodb://localhost:27017/mgr --collection lastfm_playedtracks /tmp/mgr-export/lastfm_playedtracks.json
# podman exec -i -t mongomgr mongoimport --uri=mongodb://localhost:27017/mgr --collection spotify_audiofeatures /tmp/mgr-export/spotify_audiofeatures.json

mongo_getlast:
	podman exec -i -t mongomgr mongo mgr  --eval 'db.lastfm_playedtracks.find().sort({ "$$natural": -1}).limit(1)'