namespace Api.Data
{
    public class AlarmInfo
    {
        public required DateTime LastModelTrainTime { get; set; }
        public required DateTime LastPredictionTime { get; set; }
        public Dictionary<string, object>? Metadata { get; set; }
        public required Dictionary<string, Dictionary<string, bool>> RegionsForecast { get; set; }
    }
}