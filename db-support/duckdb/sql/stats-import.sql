
COPY users (
    Id,
    Reputation,
    CreationDate,
    DisplayName,
    LastAccessDate,
    WebsiteUrl,
    Location,
    AboutMe,
    Views,
    UpVotes,
    DownVotes,
    AccountId,
    Age,
    ProfileImageUrl)
FROM 'users.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;


COPY badges (Id, UserId, Name, Date)
FROM 'badges.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY posts (
    Id,
    PostTypeId,
    AcceptedAnswerId,
    CreationDate,
    Score,
    ViewCount,
    Body,
    OwnerUserId,
    LasActivityDate,
    Title,
    Tags,
    AnswerCount,
    CommentCount,
    FavoriteCount,
    LastEditorUserId,
    LastEditDate,
    CommunityOwnedDate,
    ParentId,
    ClosedDate,
    OwnerDisplayName,
    LastEditorDisplayName)
FROM 'posts.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY tags (Id, TagName, Count, ExcerptPostId, WikiPostId)
FROM 'tags.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY postLinks (Id, CreationDate, PostId, RelatedPostId, LinkTypeId)
FROM 'postLinks.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY postHistory (
    Id,
    PostHistoryTypeId,
    PostId,
    RevisionGUID,
    CreationDate,
    UserId,
    Text,
    Comment,
    UserDisplayName)
FROM 'postHistory.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY comments (
    Id,
    PostId,
    Score,
    Text,
    CreationDate,
    UserId,
    UserDisplayName)
FROM 'comments.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

COPY votes (Id, PostId, VoteTypeId, CreationDate, UserId, BountyAmount)
FROM 'votes.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;
