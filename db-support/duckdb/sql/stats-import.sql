
.print 'importing table users'
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


.print 'importing table badges'
COPY badges (Id, UserId, Name, Date)
FROM 'badges.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

.print 'importing table posts'
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

.print 'importing table tags'
COPY tags (Id, TagName, Count, ExcerptPostId, WikiPostId)
FROM 'tags.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

.print 'importing table postLinks'
COPY postLinks (Id, CreationDate, PostId, RelatedPostId, LinkTypeId)
FROM 'postLinks.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;

.print 'importing table postHistory'
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

.print 'importing table comments'
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

.print 'importing table votes'
COPY votes (Id, PostId, VoteTypeId, CreationDate, UserId, BountyAmount)
FROM 'votes.csv'
DELIMITER ',' NULL 'NULL' CSV HEADER;
