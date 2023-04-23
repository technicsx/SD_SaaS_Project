using Microsoft.Extensions.Options;

namespace Api.Services
{
    public interface ILocationService
    {
        Task<int> GetRegionId(float lat, float lon);
        Task<int> GetRegionId(string name);
        int IntoRegionId(string regionName);
        string IntoRegionName(int regionId);
    }

    public class LocationService : ILocationService
    {
        private readonly HttpClient _client;
        private readonly PositionStackConfig _config;

        public LocationService(HttpClient client, IOptions<PositionStackConfig> options)
        {
            _client = client;
            _config = options.Value;
        }

        public async Task<int> GetRegionId(float lat, float lon)
        {
            var response =
                await _client.GetFromJsonAsync<PositionStackResponse>(
                    UrlHelpers.EncodeUrlParameters(
                        $"reverse?access_key={_config.ApiKey}&query={lat},{lon}").ToString());
            if (response is null or { Data.Count: 0 }) throw new Exception("Unable to get location");

            return IntoRegionId(response.Data[0].Region);
        }

        public async Task<int> GetRegionId(string name)
        {
            var response =
                await _client.GetFromJsonAsync<PositionStackResponse>(
                    UrlHelpers.EncodeUrlParameters(
                        $"forward?access_key={_config.ApiKey}&query={name}").ToString());
            if (response is null or { Data.Count: 0 }) throw new Exception("Unable to get location");

            return IntoRegionId(response.Data[0].Region);
        }

        public int IntoRegionId(string regionName)
        {
            return regionName switch
            {
                "Crimea" => 1,
                "Vinnytsia" => 2,
                "Volyn" => 3,
                "Dnipropetrovs'k" => 4,
                "Donetsk" => 5,
                "Zhytomyr" => 6,
                "Zakarpattia" => 7,
                "Zaporizhia" => 8,
                "Ivano-Frankivs'k" => 9,
                "Kiev" => 10,
                "Kirovohrad" => 11,
                "Luhansk" => 12,
                "Lviv" => 13,
                "Mykolaiv" => 14,
                "Odessa" => 15,
                "Poltava" => 16,
                "Rivne" => 17,
                "Sumy" => 18,
                "Ternopil" => 19,
                "Kharkiv" => 20,
                "Kherson" => 21,
                "Khmelnytskyi" => 22,
                "Cherkasy" => 23,
                "Chernivtsi" => 24,
                "Chernihiv" => 25,
                _ => throw new BadHttpRequestException($"Not supported region - {regionName}")
            };
        }

        public string IntoRegionName(int regionId)
        {
            return regionId switch
            {
                1 => "Crimea",
                2 => "Vinnytsia",
                3 => "Volyn",
                4 => "Dnipropetrovs'k",
                5 => "Donetsk",
                6 => "Zhytomyr",
                7 => "Zakarpattia",
                8 => "Zaporizhia",
                9 => "Ivano-Frankivs'k",
                10 => "Kiev",
                11 => "Kirovohrad",
                12 => "Luhansk",
                13 => "Lviv",
                14 => "Mykolaiv",
                15 => "Odessa",
                16 => "Poltava",
                17 => "Rivne",
                18 => "Sumy",
                19 => "Ternopil",
                20 => "Kharkiv",
                21 => "Kherson",
                22 => "Khmelnytskyi",
                23 => "Cherkasy",
                24 => "Chernivtsi",
                25 => "Chernihiv",
                _ => throw new ArgumentOutOfRangeException(nameof(regionId), regionId, null)
            };
        }
    }

    public class PositionStackResponse
    {
        public IList<RegionInfo> Data { get; set; } = Array.Empty<RegionInfo>();
    }

    public class RegionInfo
    {
        public float Latitude { get; set; }
        public float Longitude { get; set; }
        public string Region { get; set; } = null!;
    }

    public class PositionStackConfig
    {
        public string ApiKey { get; set; } = null!;
    }
}