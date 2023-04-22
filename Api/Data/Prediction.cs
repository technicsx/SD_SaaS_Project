namespace Api.Data
{
    public class Prediction
    {
        public int RegionId { get; set; }
        public DateTime DateHour { get; set; }
        public bool IsAlarm { get; set; }
    }
}