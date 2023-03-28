local http_request = require("http.request")
local htmlparser = require("htmlparser")
local gumbo = require("gumbo")
local lfs = require("lfs")

-- constants    (╯°□°）╯︵ ┻━┻    (╯°□°）╯︵ ┻━┻    (╯°□°）╯︵ ┻━┻
local daySecs = 86400

-- checks if folder exists by path
local function directoryExists(path)
    local attributes = lfs.attributes(path)
    return attributes and attributes.mode == "directory"
end

local folderPathParam = arg[1]
assert(folderPathParam ~= nil and directoryExists(folderPathParam), "Invalid folder is provided")

-- parses date string to os.time() format
local function parseDateTime(dateString)
    local pattern = "(%d+)-(%d+)-(%d+)"
    local year, month, day = dateString:match(pattern)
    return os.time({ year = year, month = month, day = day })
end

-- checks if date is valid
local curDateTime = os.time()
local startDateTime = parseDateTime("2022-02-24")
local fetchDateTimeEnd = #arg ~= 1 and parseDateTime(arg[2]) or nil
local fetchDateTimeStart = arg[3] ~= nil and parseDateTime(arg[3]) or parseDateTime("2022-02-24")
assert(fetchDateTimeEnd == nil or (fetchDateTimeEnd <= curDateTime and fetchDateTimeEnd >= startDateTime), "Invalid end date is provided")
assert(fetchDateTimeStart >= startDateTime and (fetchDateTimeEnd == nil or fetchDateTimeStart <= fetchDateTimeEnd), "Invalid start date is provided")


-- fetches article by uri
local function fetchArticle(uri)
    local headers, stream = assert(http_request.new_from_uri(uri):go())

    local body = assert(stream:get_body_as_string())

    if headers:get ":status" == "404" then
        return nil
    elseif headers:get ":status" ~= "200" then
        error("Code: " .. headers:get ":status")
    end

    return body
end

-- selects article body from htmlparser
local function selectArticleBodyByAttribute(body)
    local htmlTreeRoot = htmlparser.parse(body, 2000)
    local divs = htmlTreeRoot:select("[property='content:encoded']")
    local parsedBody = ""
    for _, div in ipairs(divs) do
        local text = div:getcontent()
        parsedBody = parsedBody .. "\n" .. text
    end
    return parsedBody
end

-- parses article body and removes unnecessary tags or elements
local function parse(body)
    -- parse html
    local document = assert(gumbo.parse(body), "Failed to parse html")

    -- remove all links
    local aTags = document.links
    for _, element in ipairs(aTags) do
        element:remove()
    end

    -- remove all images
    local imgTags = document.images
    for _, element in ipairs(imgTags) do
        element:remove()
    end

    -- remove all spans with style="text-decoration: underline;"
    local underlinedSpans = document:getElementsByTagName("span")
    for _, span in ipairs(underlinedSpans) do
        if span ~= nil and span:getAttribute("style") == "text-decoration: underline;" then
            span:remove()
        end
    end

    -- remove all hr tags with align="left" size="1" width="33%" with different behaviour depending on attribute size
    local hrs = document:getElementsByTagName("hr")
    for _, hr in ipairs(hrs) do
        if hr ~= nil and hr:getAttribute("align") == "left" and hr:getAttribute("size") == "1" and hr:getAttribute("width") == "33%" then
            local currentNode = hr.parentNode
            while currentNode do
                local nextNode = currentNode.nextSibling
                currentNode:remove()
                currentNode = nextNode
            end
        elseif hr ~= nil and hr:getAttribute("align") == "left" and hr:getAttribute("size") == "2" and hr:getAttribute("width") == "33%" then
            local currentNode = hr.nextSibling
            while currentNode do
                local nextNode = currentNode.nextSibling
                currentNode:remove()
                currentNode = nextNode
            end
            hr:remove()
        end
    end



    local ps = document:getElementsByTagName("p")
    -- remove all p tags depending on their attributes or position
    for _, p in ipairs(ps) do
        if p ~= nil and p:getAttribute("align") == "center" and not (p.firstElementChild ~= nil and p.firstElementChild:getAttribute("style") == "font-size: 13.008px;") then
            p:remove()
        end

        if p ~= nil and p.innerHTML == '<p style="text-align: left;">&nbsp;</p>' then
            local currentNode = p.nextSibling
            while currentNode do
                local nextNode = currentNode.nextSibling
                currentNode:remove()
                currentNode = nextNode
            end
            p:remove()
        end

        if p ~= nil and p.innerHTML:match("pm ET") then
            local currentNode = p.previousSibling
            while currentNode do
                local nextNode = currentNode.previousSibling
                currentNode:remove()
                currentNode = nextNode
            end
            p:remove()
        end

        local pStr = p.innerHTML
        if p ~= nil and (pStr:match("Key Takeaways") or pStr:match("Immediate items to watch") or pStr:match("Click") or pStr:match("Note:") or p.outerHTML:match("<p>%[%d+%]&nbsp;")) then
            p:remove()
        end
    end

    -- remove all divs that contain "Click" in their outerHTML
    local divs = document:getElementsByTagName("div")
    for _, div in ipairs(divs) do
        local divStr = div.outerHTML
        if div ~= nil and (divStr:match("Click")) then
            div:remove()
        end
    end

    return document.body.textContent
    -- return document:serialize()
end

-- writes article to folder path
local function writeToFile(filename, folderPath, filecontent)
    lfs.mkdir(folderPath)

    local fullFileName = folderPath .. "/" .. filename .. ".txt"
    print("Writing to: " .. fullFileName)
    local file, err = io.open(fullFileName, "w")

    if not file then
        print("Error writing to file:", err)
    else
        local metaString = string.gsub(filename, "assessment%-", "")
        file:write(metaString .. "\n" .. filecontent)
        file:close()
    end
end

-- forms isw article uri
local function formArticleUri(dateTime)
    local date, _, _ = os.date("*t", dateTime)

    local uriBase = "https://www.understandingwar.org/backgrounder"

    ---@diagnostic disable-next-line: param-type-mismatch
    local curDateString = string.gsub(os.date('%B-%e-%Y', dateTime), "%s+", "")
    local formattedDateWithYear = string.lower(string.sub(curDateString, 1, 1)) .. string.sub(curDateString, 2)

    ---@diagnostic disable-next-line: param-type-mismatch
    local curDateStringWithoutYear = string.gsub(os.date('%B-%e', dateTime), "%s+", "")
    local formattedDateWithoutYear = string.lower(string.sub(curDateStringWithoutYear, 1, 1)) ..
        string.sub(curDateStringWithoutYear, 2)

    -- ...
    if date.year == 2022 then
        if date.month == 2 then
            if date.day == 24 then
                return uriBase .. "/russia-ukraine-warning-update-initial-russian-offensive-campaign-assessment"
            elseif date.day == 25 then
                return uriBase .. "/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-25-2022"
            elseif date.day == 26 then
                return uriBase .. "/russia-ukraine-warning-update-russian-offensive-campaign-assessment-february-26"
            elseif date.day == 27 then
                return nil
            elseif date.day == 28 then
                return uriBase .. "/russian-offensive-campaign-assessment-february-28-2022"
            end
        elseif date.month == 5 and date.day == 5 then
            return uriBase .. "/russian-campaign-assessment-may-5"
        elseif date.month == 7 and date.day == 11 then
            return uriBase .. "/russian-offensive-campaign-update-july-11"
        elseif date.month == 8 and date.day == 12 then
            return uriBase .. "/russian-offensive-campaign-assessment-august-12-0"
        elseif date.month == 11 and date.day == 24 then
            return nil
        elseif date.month == 12 and date.day == 25 then
            return nil
        end

        return uriBase .. "/russian-offensive-campaign-assessment-" .. formattedDateWithoutYear
    elseif date.year == 2023 then
        if date.month == 1 and date.day == 1 then
            return nil
        elseif date.month == 2 and date.day == 5 then
            return uriBase .. "/russian-offensive-campaign-update-february-5-2023"
        end

        return uriBase .. "/russian-offensive-campaign-assessment-" .. formattedDateWithYear
    end

    return nil
end

-- main function
local function run(dateTime, forceExitOnError)
    local date, _, _ = os.date("*t", dateTime)
    local filename = string.format("assessment-%04d-%02d-%02d", date.year, date.month, date.day)

    local articleUri = formArticleUri(dateTime)
    if (articleUri == nil) then
        print("Skipping " .. filename .. ': missing article')
        if forceExitOnError then os.exit(7) end

        return
    end

    local rawHTML = fetchArticle(articleUri)
    if (rawHTML == nil) then
        print("Skipping " .. filename .. ': got 404 for article with the given date ')
        if forceExitOnError then os.exit(7) end

        return
    end

    local articleBodyString = selectArticleBodyByAttribute(rawHTML)

    local stringToWrite = parse(articleBodyString)

    local folderToWriteTo = folderPathParam .. '/' .. os.date("%Y-%m", dateTime)

    writeToFile(filename, folderToWriteTo, stringToWrite)
end

-- script actions
if fetchDateTimeEnd ~= nil then
    assert(fetchDateTimeStart ~= nil and fetchDateTimeEnd ~= nil, "Both start and end dates must be specified")
    for dateTime = fetchDateTimeStart, fetchDateTimeEnd, daySecs do
        run(dateTime, false)
    end
else
    run(os.time() - daySecs, true)
end
