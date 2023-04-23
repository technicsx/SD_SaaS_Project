namespace Api
{
    public static class DateTimeExtensions
    {
        public static DateTime RoundToFloorHour(this DateTime date)
        {
            return new DateTime(date.Year, date.Month, date.Day, date.Hour, 0, 0, 0, date.Kind);
        }
    }
}