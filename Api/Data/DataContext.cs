using Microsoft.EntityFrameworkCore;

namespace Api.Data
{
    public class DataContext : DbContext
    {
        public DataContext(DbContextOptions<DataContext> options) : base(options)
        {
        }

        public DbSet<Prediction> Predictions { get; set; } = null!;
        public DbSet<Metadata> Metadata { get; set; } = null!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Prediction>(e =>
            {
                e.HasKey(p => new { p.RegionId, p.DateHour });
                e.HasData(GetPredictionTestData());
                e.Property(p => p.DateHour)
                    .HasConversion(v => v, 
                        v => DateTime.SpecifyKind(v, DateTimeKind.Utc));
            });

            modelBuilder.Entity<Metadata>(e =>
            {
                e.HasKey(m => m.Key);
                e.HasData(GetTestMetadata());
            });
        }

        private static IEnumerable<Prediction> GetPredictionTestData()
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
        
        private static Metadata[] GetTestMetadata()
        {
            return new[]
            {
                new Metadata
                {
                    Key = nameof(AlarmInfo.LastModelTrainTime),
                    Value = new DateTime(2023, 5, 24).ToString("s")
                },
                new Metadata
                {
                    Key = nameof(AlarmInfo.LastPredictionTime),
                    Value = new DateTime(2023, 5, 24).ToString("s")
                }
            };
        }
    }
}