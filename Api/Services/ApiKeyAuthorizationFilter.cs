namespace Api.Services
{
    public class ApiKeyAuthorizationFilter : IEndpointFilter
    {
        private const string ApiKeyHeaderName = "X-API-Key";

        private readonly IApiKeyValidator _apiKeyValidator;

        public ApiKeyAuthorizationFilter(IApiKeyValidator apiKeyValidator)
        {
            _apiKeyValidator = apiKeyValidator;
        }

        public async ValueTask<object?> InvokeAsync(EndpointFilterInvocationContext context,
            EndpointFilterDelegate next)
        {
            var apiKey = context.HttpContext.Request.Headers[ApiKeyHeaderName];

            if (await _apiKeyValidator.IsValid(apiKey))
            {
                return await next(context);
            }

            return Results.Unauthorized();
        }
    }
}