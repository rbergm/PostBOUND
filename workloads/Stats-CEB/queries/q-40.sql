SELECT COUNT(*)
FROM postLinks AS pl,
  posts AS p,
  users AS u,
  badges AS b
WHERE p.Id = pl.RelatedPostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND pl.LinkTypeId = 1
  AND p.Score >= -1
  AND p.CommentCount <= 8
  AND p.CreationDate >= CAST('2010-07-21 12:30:43' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-07 01:11:03' AS timestamp)
  AND u.Views <= 40
  AND u.CreationDate >= CAST('2010-07-26 19:11:25' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-11 22:26:42' AS timestamp);
