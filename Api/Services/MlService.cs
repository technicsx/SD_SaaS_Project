namespace Api.Services
{
    public interface IMlService
    {
        Task<string> UpdatePredictions();
    }

    public class MlService : IMlService
    {
        private readonly HttpClient _client;

        public MlService(HttpClient client)
        {
            _client = client;
        }

        public async Task<string> UpdatePredictions()
        {
            var response = await _client.PostAsync(new Uri("api/update-predictions", UriKind.Relative), null);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}