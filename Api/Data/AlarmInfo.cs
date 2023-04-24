namespace Api.Data
{
    public class AlarmInfo
    {
        public required string LastModelTrainTime { get; set; }
        public required string LastPredictionTime { get; set; }
        public required Dictionary<string, Dictionary<string, bool>> RegionsForecast { get; set; }
    }
}