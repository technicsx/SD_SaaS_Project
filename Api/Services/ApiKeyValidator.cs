using Microsoft.Extensions.Options;

namespace Api.Services
{
    public interface IApiKeyValidator
    {
        Task<bool> IsValid(string? apiKey);
    }
    
    public class ApiKeyValidator : IApiKeyValidator
    {
        private readonly IOptions<AuthConfig> _config;

        public ApiKeyValidator(IOptions<AuthConfig> config)
        {
            _config = config;
        }

        public Task<bool> IsValid(string? apiKey)
        {
            return Task.FromResult(apiKey == _config.Value.ApiKey);
        }
    }

    public class AuthConfig
    {
        public string? ApiKey { get; set; }
    }
}