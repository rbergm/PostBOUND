SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  users AS u
WHERE pl.RelatedPostId = p.Id
  AND u.Id = c.UserId
  AND c.PostId = p.Id
  AND ph.PostId = p.Id
  AND c.CreationDate >= CAST('2010-07-11 12:25:05' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-11 13:43:09' AS timestamp)
  AND p.CommentCount >= 0
  AND p.CommentCount <= 14
  AND pl.LinkTypeId = 1
  AND ph.CreationDate >= CAST('2010-08-06 03:14:53' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 491
  AND u.DownVotes >= 0
  AND u.DownVotes <= 0;
