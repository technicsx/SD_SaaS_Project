using Api.Data;
using Microsoft.EntityFrameworkCore;

namespace Api.Services
{
    public interface IPredictionService
    {
        Task<IList<Prediction>> GetPredictions(int regionId, DateTime dateHour, int take = 12);
    }

    public class PredictionService : IPredictionService
    {
        private readonly DataContext _db;

        public PredictionService(DataContext db)
        {
            _db = db;
        }

        public async Task<IList<Prediction>> GetPredictions(int regionId, DateTime dateHour, int take = 12)
        {
            var firstHour = dateHour.RoundToFloorHour();
            var lastHour = firstHour + TimeSpan.FromHours(take);

            return await _db.Predictions
                .Where(r => r.RegionId == regionId
                            && r.DateHour >= firstHour && r.DateHour < lastHour)
                .ToListAsync();
        }
    }
}