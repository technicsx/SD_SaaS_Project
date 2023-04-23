using Api.Data;
using Api.Services;
using AspNetCore.Authentication.ApiKey;
using Microsoft.AspNetCore.Authorization;
using Microsoft.OpenApi.Models;

var builder = WebApplication.CreateBuilder(args);
var services = builder.Services;

#region DI

services.AddSqlite<DataContext>(builder.Configuration.GetConnectionString("SQLite"));
services.Configure<PositionStackConfig>(builder.Configuration.GetSection("PositionStack"));
services.AddHttpClient<ILocationService, LocationService>(client =>
{
    client.BaseAddress = new Uri("http://api.positionstack.com/v1/", UriKind.Absolute);
});
services.AddScoped<IPredictionService, PredictionService>();

services.Configure<AuthConfig>(builder.Configuration.GetSection("Auth"));
services.AddAuthentication(ApiKeyDefaults.AuthenticationScheme)
    .AddApiKeyInHeaderOrQueryParams<ApiKeyProvider>(options =>
    {
        options.Realm = "Alarms Web API";
        options.KeyName = "X-API-KEY";
    });
services.AddAuthorization(options =>
{
    options.DefaultPolicy = new AuthorizationPolicyBuilder()
        .RequireAuthenticatedUser()
        .AddAuthenticationSchemes(ApiKeyDefaults.AuthenticationScheme)
        .Build();
});

services.AddEndpointsApiExplorer();
services.AddSwaggerGen(c =>
{
    c.AddSecurityDefinition("Auth-Header", new OpenApiSecurityScheme
    {
        Description = "Please provide your API key",
        In = ParameterLocation.Header,
        Name = "X-API-Key",
        Type = SecuritySchemeType.ApiKey,
        Scheme = ApiKeyDefaults.AuthenticationScheme
    });

    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {  new OpenApiSecurityScheme
        {
            Name = "X-API-Key",
            In = ParameterLocation.Header,
            Reference = new OpenApiReference
            {
                Id = "Auth-Header",
                Type = ReferenceType.SecurityScheme
            }
        }, Array.Empty<string>() }
    });
});

#endregion


var app = builder.Build();

app.UseAuthentication();
app.UseAuthorization();

#region Routes

app.MapGet("/", () => "Hello World!");

var api = app.MapGroup("/api").RequireAuthorization();

api.MapGet("prediction",
    async ([AsParameters] QueryParams query, ILocationService locationService, IPredictionService predictionService) =>
    {
        int? regionId = query switch
        {
            { RegionId: { } id } => id,
            { Name: { } name } => await locationService.GetRegionId(name),
            { Lat: { } lat, Lon: { } lon } => await locationService.GetRegionId(lat, lon),
            _ => null
        };

        return await predictionService.GetAlarmInfo(regionId, DateTime.UtcNow);
    });

#endregion

app.UseSwagger();
app.UseSwaggerUI();

using (var scope = app.Services.CreateScope())
{
    var scopeServices = scope.ServiceProvider;

    var context = scopeServices.GetRequiredService<DataContext>();
    await context.Database.EnsureCreatedAsync();
}

app.Run();

public class QueryParams
{
    public int? RegionId { get; set; }
    public string? Name { get; set; }
    public float? Lat { get; set; }
    public float? Lon { get; set; }
}