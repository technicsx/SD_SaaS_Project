using Microsoft.EntityFrameworkCore;

namespace Api.Data
{
    public class DataContext : DbContext
    {
        public DataContext(DbContextOptions<DataContext> options) : base(options)
        {
        }

        public DbSet<Prediction> Predictions { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Prediction>(e =>
            {
                e.HasKey(p => new { p.RegionId, p.DateHour });
                e.HasData(GetTestData());
            });
        }

        private IEnumerable<Prediction> GetTestData()
        {
            var now = DateTime.UtcNow;
            var nowRounded = now.RoundToFloorHour();

            for (int regionId = 1; regionId <= 25; regionId++)
            {
                for (int hour = 0; hour < 12; hour++)
                {
                    var nextTime = nowRounded + TimeSpan.FromHours(hour);
                    yield return new Prediction
                    {
                        DateHour = nextTime,
                        IsAlarm = Random.Shared.Next(2) > 0,
                        RegionId = regionId
                    };
                }
            }
        }
    }
}