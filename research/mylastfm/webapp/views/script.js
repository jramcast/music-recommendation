const labels = [
    'Acousticness',
    'Danceability',
    'Instrumentalness',
    'Energy',
    'Valence'
]

const options = {
    maintainAspectRatio: false,
    elements: {
        line: {
            borderWidth: 3
        }
    },
    scales: {
        r: {
            axis: 'r',
            min: 0,
            max: 1
        }
    }
};

window.addEventListener('load', () => {
    const canvasElements = document.querySelectorAll(".track-features-chart");
    const recommendationElement = document.getElementById("recommendation");
    const recommendation = JSON.parse(recommendationElement.dataset.recommendation);
    canvasElements.forEach(element => {
        const data = buildDataSetFromElement(element, recommendation);
        new Chart(element, {
            type: 'radar',
            data,
            options
        });
    });
});

function buildDataSetFromElement(element, recommendation) {
    const recommendedTrack = JSON.parse(element.dataset.recommendedTrack);
    const { track } = recommendedTrack;
    const title = `${track.artist} - ${track.trackName}`;
    const {
        acousticness, danceability, instrumentalness, energy, valence
    } = recommendedTrack.track.audiofeatures;

    return {
        labels,
        datasets: [{
            label: title,
            data: [
                acousticness,
                danceability,
                instrumentalness,
                energy,
                valence
            ],
            fill: true,
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgb(255, 99, 132)',
            pointBackgroundColor: 'rgb(255, 99, 132)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(255, 99, 132)'
        }, {
            label: 'Inferred features',
            data: [
                recommendation.features.acousticness,
                recommendation.features.danceability,
                recommendation.features.instrumentalness,
                recommendation.features.energy,
                recommendation.features.valence
            ],
            fill: true,
            backgroundColor: 'rgba(64, 172, 245, 0.2)',
            borderColor: 'rgb(64, 172, 245)',
            pointBackgroundColor: 'rgb(64, 172, 245)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(54, 172, 245)'
        }]
    };

}