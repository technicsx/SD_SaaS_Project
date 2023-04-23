using Api.Data;
using Api.Services;

var builder = WebApplication.CreateBuilder(args);
var services = builder.Services;

services.AddSqlite<DataContext>(builder.Configuration.GetConnectionString("SQLite"));
services.Configure<PositionStackConfig>(builder.Configuration.GetSection("PositionStack"));
services.AddHttpClient<ILocationService, LocationService>(client =>
{
    client.BaseAddress = new Uri("http://api.positionstack.com/v1/", UriKind.Absolute);
});
services.AddScoped<IPredictionService, PredictionService>();
services.AddEndpointsApiExplorer();
services.AddSwaggerGen();

var app = builder.Build();

app.MapGet("/", () => "Hello World!");

app.MapGet("/api/prediction",
    async ([AsParameters] QueryParams query, ILocationService locationService, IPredictionService predictionService) =>
    {
        var regionId = query switch
        {
            { RegionId: { } id } => id,
            { Name: { } name } => await locationService.GetRegionId(name),
            { Lat: { } lat, Lon: { } lon } => await locationService.GetRegionId(lat, lon),
            _ => throw new BadHttpRequestException("Please provide region id, location name or coordinates.")
        };

        return await predictionService.GetPredictions(regionId, DateTime.UtcNow);
    });

using (var scope = app.Services.CreateScope())
{
    var scopeServices = scope.ServiceProvider;

    var context = scopeServices.GetRequiredService<DataContext>();
    await context.Database.EnsureCreatedAsync();
}

app.UseSwagger();
app.UseSwaggerUI();

app.Run();

public class QueryParams
{
    public int? RegionId { get; set; }
    public string? Name { get; set; }
    public float? Lat { get; set; }
    public float? Lon { get; set; }
}