using System.Security.Claims;
using AspNetCore.Authentication.ApiKey;
using Microsoft.Extensions.Options;

namespace Api.Services
{
    public class ApiKeyProvider : IApiKeyProvider
    {
        private readonly IOptions<AuthConfig> _config;

        public ApiKeyProvider(IOptions<AuthConfig> config)
        {
            _config = config;
        }

        public Task<IApiKey?> ProvideAsync(string key)
        {
            if (key == _config.Value.ApiKey)
            {
                return Task.FromResult<IApiKey?>(new ApiKey()
                {
                    Key = key
                });
            }

            return Task.FromResult<IApiKey?>(null);
        }
    }

    public class ApiKey : IApiKey
    {
        public required string Key { get; init; }
        public string OwnerName { get; init; } = "Alarms Api Client";
        public IReadOnlyCollection<Claim> Claims { get; init; } = Array.Empty<Claim>();
    }
    
    public class AuthConfig
    {
        public string? ApiKey { get; set; }
    }
}