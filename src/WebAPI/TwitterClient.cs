using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

namespace WebAPI
{
    public class TwitterClient : IDisposable
    {

        public TwitterClient(string bearerToken)
        {
            _httpClient = new();
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);

            _jsonOptions = new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        }

        public async Task<List<TweetDetail>> GetRecentTweets(string query)
        {
            var res = await _httpClient.GetAsync(_baseUrl + "tweets/search/recent?query=" + query);
            var json = await res.Content.ReadAsStringAsync();

            return ParseData<List<TweetDetail>>(json);
        }

        private T ParseData<T>(string json)
        {
            var answer = JsonSerializer.Deserialize<TwitterResponseModel<T>>(json, _jsonOptions);
            return answer.Data;
        }

        private const string _baseUrl = "https://api.twitter.com/2/";

        private readonly HttpClient _httpClient;
        private readonly JsonSerializerOptions _jsonOptions;

        public void Dispose()
        {
            _httpClient?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
    public class TwitterResponseModel<T>
    {
        public T Data { set; get; }

        public string Detail { init; get; }
        public string Title { init; get; }
        public string Type { init; get; }
    }

    public class TweetDetail
    {
        public string Id { init; get; }
        public string Text { init; get; }
    }
}
