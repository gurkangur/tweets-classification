using DataClassification;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class TweetsController : Controller
    {
        private readonly PredictionEnginePool<Tweet, TweetPrediction> _predictionEnginePool;

        public TweetsController(PredictionEnginePool<Tweet, TweetPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }


        [HttpGet("recent")]
        public async Task<IEnumerable<Tweet>> GetRecentTweets([FromQuery(Name = "query")] string query)
        {

            var client = new TwitterClient("bearerToken");
            var tweets = await client.GetRecentTweets(query);

            return tweets.Select(t => new Tweet
            {
                Id = t.Id,
                FullText = t.Text,
                Category = _predictionEnginePool.Predict(new Tweet { FullText = t.Text }).Category
            });
        }
    }
}
