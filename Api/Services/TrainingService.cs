namespace Api.Services
{
    public interface ITrainingService
    {
        Task TrainModel();
    }

    public class TrainingService : ITrainingService
    {
        private readonly HttpClient _client;

        public TrainingService(HttpClient client)
        {
            _client = client;
        }

        public async Task TrainModel()
        {
            var response = await _client.PostAsync(new Uri("api/training", UriKind.Relative), null);
            response.EnsureSuccessStatusCode();
        }
    }
}