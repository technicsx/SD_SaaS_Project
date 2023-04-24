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
            await _client.PostAsync(new Uri("/api/training"), null);
        }
    }
}