SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  users AS u
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.OwnerUserId = u.Id
  AND c.CreationDate >= CAST('2010-07-21 11:05:37' AS timestamp)
  AND c.CreationDate <= CAST('2014-08-25 17:59:25' AS timestamp)
  AND u.UpVotes >= 0
  AND u.CreationDate >= CAST('2010-08-21 21:27:38' AS timestamp);
