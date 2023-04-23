using Api.Data;
using Microsoft.EntityFrameworkCore;

namespace Api.Services
{
    public interface IPredictionService
    {
        Task<AlarmInfo> GetAlarmInfo(int? regionId, DateTime dateHour, int take = 12);
    }

    public class PredictionService : IPredictionService
    {
        private readonly DataContext _db;
        private readonly ILocationService _locationService;

        public PredictionService(DataContext db, ILocationService locationService)
        {
            _db = db;
            _locationService = locationService;
        }

        public async Task<AlarmInfo> GetAlarmInfo(int? regionId, DateTime dateHour, int take = 12)
        {
            var firstHour = dateHour.RoundToFloorHour();
            var lastHour = firstHour + TimeSpan.FromHours(take);

            var query = _db.Predictions
                .OrderBy(p => p.DateHour)
                .Where(p => p.DateHour >= firstHour && p.DateHour < lastHour);
            if (regionId.HasValue)
                query = query.Where(p => p.RegionId == regionId);

            var result = await query.ToListAsync();

            return new AlarmInfo
            {
                LastModelTrainTime = new DateTime(2023, 4, 24),
                LastPredictionTime = DateTime.Now,
                RegionsForecast = result.GroupBy(r => r.RegionId, (id, predictions) =>
                {
                    return new
                    {
                        Id = id,
                        Predictions = predictions.ToDictionary(p => $"{p.DateHour:hh:mm}", p => p.IsAlarm)
                    };
                }).ToDictionary(g => _locationService.IntoRegionName(g.Id), g => g.Predictions)
            };
        }
    }
}